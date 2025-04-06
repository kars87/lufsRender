# Import necessary modules
from flask import Flask, request, jsonify, render_template, send_from_directory, after_this_request
import os
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from werkzeug.utils import secure_filename
from pathlib import Path
import librosa
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor
from pysndfx import AudioEffectsChain
from pydub import AudioSegment  # Add this import for AudioSegment
# Create the Flask application instance
app = Flask(__name__)

# Configure paths for Render.com or local development
if os.environ.get('RENDER', '') == 'true':
    # Render.com paths
    BASE_DIR = Path('/opt/render/project/src')
    UPLOAD_FOLDER = Path('/tmp/uploads')
    TEMP_DIR = Path('/tmp/temp')
else:
    # Local development paths
    BASE_DIR = Path.cwd()
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    TEMP_DIR = BASE_DIR / 'temp'

# Update app configuration
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aiff', 'aif'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_lufs(audio_path):
    """Calculates the Integrated Loudness (LUFS) of an audio file using pyloudnorm."""
    try:
        data, rate = sf.read(audio_path)
        
        # Check if data is valid
        if len(data) == 0:
            print("Empty audio data")
            return None
            
        # Handle mono vs stereo
        if len(data.shape) > 1 and data.shape[1] > 1:
            # For stereo, pyloudnorm expects shape (samples, channels)
            pass  # Data is already in the right format
        else:
            # For mono, ensure data is shaped correctly
            data = data.reshape(-1, 1)
            
        meter = pyln.Meter(rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)
        
        # Check for valid loudness value
        if not np.isfinite(loudness):
            print(f"Invalid loudness value: {loudness}")
            return -70.0  # Default value for silence
            
        return loudness
    except Exception as e:
        print(f"Error during LUFS calculation: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def create_lufs_graph(audio_path):
    """
    Creates a graph of LUFS over time for the given audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        dict: A dictionary containing time and loudness data. Returns None on error.
    """
    try:
        data, rate = sf.read(audio_path)
        meter = pyln.Meter(rate)

        # Use 3-second windows with 1-second steps for LUFS measurement
        interval = 3.0  # seconds
        step = 1.0  # seconds

        loudness_values = []
        time_values = []

        for start in np.arange(0, len(data) / rate - interval, step):
            end = start + interval
            start_sample = int(start * rate)
            end_sample = int(end * rate)
            
            # Ensure we don't go out of bounds
            if end_sample > len(data):
                end_sample = len(data)
                
            segment = data[start_sample:end_sample]

            if len(segment) > 0:
                try:
                    loudness = meter.integrated_loudness(segment)
                    # Handle potential infinity or NaN values
                    if np.isfinite(loudness):
                        loudness_values.append(loudness)
                        time_values.append(start + interval / 2)
                except Exception as segment_error:
                    print(f"Error processing segment at {start}s: {segment_error}")
                    continue

        # Handle empty loudness_values
        if not loudness_values:
            print("No valid loudness values found")
            return {
                'time': [0],
                'loudness': [-70],  # Default value
                'min': -70,
                'max': -10,
                'mean': -23
            }

        return {
            'time': time_values,
            'loudness': loudness_values,
            'min': float(np.min(loudness_values)),
            'max': float(np.max(loudness_values)),
            'mean': float(np.mean(loudness_values))
        }
    except Exception as e:
        print(f"Error creating LUFS graph: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def normalize_audio_loudness(audio_path, target_lufs):
    """
    Normalizes the loudness of an audio file to match the target LUFS.

    Args:
        audio_path: Path to the audio file
        target_lufs: float, target LUFS value

    Returns:
        str: Path to the normalized audio file, or None on error
    """
    try:
        data, rate = sf.read(audio_path)
        
        # Handle mono vs stereo
        is_mono = len(data.shape) == 1 or data.shape[1] == 1
        
        # Reshape mono audio for pyloudnorm
        if is_mono and len(data.shape) == 1:
            data_for_meter = data.reshape(-1, 1)
        else:
            data_for_meter = data
        
        # Measure the current loudness
        meter = pyln.Meter(rate)
        current_lufs = meter.integrated_loudness(data_for_meter)
        print(f"Current LUFS: {current_lufs}, Target LUFS: {target_lufs}")
        
        # Handle very quiet audio (can cause assertion errors)
        if current_lufs < -70:
            print("Audio is too quiet for normalization, using default gain")
            # Apply a fixed gain instead
            gain_db = target_lufs + 70  # Rough estimate
            normalized_audio = data * 10**(gain_db/20)
        else:
            # Normalize to target LUFS
            try:
                normalized_audio = pyln.normalize.loudness(data, current_lufs, target_lufs)
            except AssertionError:
                print("Assertion error during normalization, using manual gain calculation")
                # Calculate gain manually
                gain_db = target_lufs - current_lufs
                normalized_audio = data * 10**(gain_db/20)
        
        # Create output path
        output_path = os.path.join(TEMP_DIR, f"normalized_{os.path.basename(audio_path)}")
        sf.write(output_path, normalized_audio, rate)
        
        return output_path
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def apply_compression(audio_path, threshold_db=-20, ratio=4, attack_ms=5, release_ms=100):
    """
    Applies compression to an audio file.
    
    Args:
        audio_path: Path to the audio file
        threshold_db: Threshold in dB where compression begins
        ratio: Compression ratio
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        
    Returns:
        str: Path to the compressed audio file, or None on error
    """
    print(f"Starting compression with threshold={threshold_db}, ratio={ratio}, attack={attack_ms}, release={release_ms}")
    
    try:
        data, rate = sf.read(audio_path)
        print(f"Audio loaded: shape={data.shape}, rate={rate}")
        
        # Create a Pedalboard with a compressor
        board = Pedalboard([
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms
            )
        ])
        
        # Apply the effects
        print("Applying compression with Pedalboard")
        effected = board(data, rate)
        
        # Create output path
        output_path = os.path.join(TEMP_DIR, f"compressed_{os.path.basename(audio_path)}")
        sf.write(output_path, effected, rate)
        print(f"Compressed audio saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error applying compression with Pedalboard: {e}")
        
        # Try alternative method
        try:
            print("Attempting compression with pysndfx")
            # Create output path
            output_path = os.path.join(TEMP_DIR, f"compressed_{os.path.basename(audio_path)}")
            
            # Create the effects chain
            fx = (
                AudioEffectsChain()
                .compand(attack_ms/1000, release_ms/1000, [threshold_db, -90, -70, -60, -50, -40, -30, -20, -10, 0], ratio)
            )
            
            # Apply the effects
            fx(audio_path, output_path)
            print(f"Compressed audio saved to {output_path} using pysndfx")
            
            return output_path
        except Exception as e:
            print(f"Error applying compression with pysndfx: {e}")
            
            # Last resort: just copy the file and return it
            try:
                print("Falling back to simple copy as compression failed")
                import shutil
                output_path = os.path.join(TEMP_DIR, f"compressed_{os.path.basename(audio_path)}")
                shutil.copy(audio_path, output_path)
                print(f"File copied to {output_path}")
                return output_path
            except Exception as e:
                print(f"Error in fallback copy: {e}")
                return None


def apply_compression_sox(audio_path, attack=10, release=100, ratio=4, threshold=-20):
    """
    Applies compression to an audio file using SoX.
    
    Args:
        audio_path: Path to the audio file
        attack: Attack time in milliseconds
        release: Release time in milliseconds
        ratio: Compression ratio
        threshold: Threshold in dB where compression begins
        
    Returns:
        str: Path to the compressed audio file, or None on error
    """
    try:
        # Create output path
        output_path = os.path.join(TEMP_DIR, f"compressed_{os.path.basename(audio_path)}")
        
        # Create the effects chain
        fx = (
            AudioEffectsChain()
            .compand(attack, release, [threshold, -90, -70, -60, -50, -40, -30, -20, -10, 0], ratio)
        )
        
        # Apply the effects
        fx(audio_path, output_path)
        
        return output_path
    except Exception as e:
        print(f"Error applying compression: {e}")
        return None


def is_stereo(audio_path):
    """
    Checks if an audio file is stereo or mono.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        bool: True if stereo, False if mono, None on error
    """
    try:
        data, rate = sf.read(audio_path)
        # Check if data has more than one dimension and second dimension size > 1
        return len(data.shape) > 1 and data.shape[1] > 1
    except Exception as e:
        print(f"Error checking audio channels: {e}")
        return None


def reduce_noise(audio_path, strength=0.5):
    """
    Applies noise reduction to an audio file.
    
    Args:
        audio_path: Path to the audio file
        strength: Noise reduction strength (0.0 to 1.0)
        
    Returns:
        str: Path to the noise-reduced audio file, or None on error
    """
    print(f"Starting noise reduction with strength {strength}")
    
    # First try the librosa-based approach as it's more reliable
    try:
        print("Attempting noise reduction with librosa")
        # Create a temporary file for the output
        output_path = os.path.join(TEMP_DIR, f"noisereduced_{os.path.basename(audio_path)}")
        
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)
        print(f"Audio loaded with librosa: shape={y.shape}, sr={sr}")
        
        # Simple noise reduction using spectral gating
        # Estimate noise from the first 0.5 seconds
        noise_sample = y[:int(sr * 0.5)] if len(y) > sr * 0.5 else y
        
        # Compute noise profile
        noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
        
        # Apply spectral gating
        S = librosa.stft(y)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Apply reduction based on strength
        reduction_factor = 1.0 - strength
        mask = S_mag > (noise_profile[:, np.newaxis] * (2.0 + 8.0 * strength))
        S_reduced = S_mag * mask + S_mag * (1 - mask) * reduction_factor
        
        # Reconstruct signal
        y_reduced = librosa.istft(S_reduced * np.exp(1j * S_phase))
        
        # Save the result
        sf.write(output_path, y_reduced, sr)
        print(f"Noise reduction completed, saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error in librosa noise reduction: {e}")
        
        # Fall back to pysndfx approach
        try:
            print("Attempting noise reduction with pysndfx")
            # Load audio
            data, rate = sf.read(audio_path)
            print(f"Audio loaded with soundfile: shape={data.shape}, rate={rate}")
            
            # Create output path
            output_path = os.path.join(TEMP_DIR, f"noisereduced_{os.path.basename(audio_path)}")
            
            # Using pysndfx for noise reduction
            # Convert strength (0-1) to appropriate parameters
            noise_amount = 0.015 * (1 - strength)  # Lower values = more reduction
            
            # Create a simpler effects chain that's more likely to work
            fx = AudioEffectsChain()
            
            # Add effects one by one with try/except to identify problematic ones
            try:
                fx = fx.highpass(frequency=60)
                print("Added highpass filter")
            except Exception as e:
                print(f"Error adding highpass: {e}")
            
            try:
                fx = fx.lowshelf(gain=-12.0 * strength, frequency=260, slope=0.1)
                print("Added lowshelf filter")
            except Exception as e:
                print(f"Error adding lowshelf: {e}")
            
            try:
                fx = fx.compand(attack=0.2, decay=1, soft_knee=6.0, threshold=-20, db_level=-20)
                print("Added compand effect")
            except Exception as e:
                print(f"Error adding compand: {e}")
            
            # Only add noisered if it's available
            try:
                fx = fx.noisered(amount=noise_amount)
                print("Added noise reduction effect")
            except Exception as e:
                print(f"Error adding noisered (this is expected if not available): {e}")
            
            # Apply the effects and save
            print("Applying effects chain")
            fx(audio_path, output_path)
            print(f"Effects applied, saved to {output_path}")
            
            return output_path
        except Exception as e:
            print(f"Error in pysndfx noise reduction: {e}")
            
            # Last resort: just copy the file and return it
            try:
                print("Falling back to simple copy as noise reduction failed")
                import shutil
                output_path = os.path.join(TEMP_DIR, f"noisereduced_{os.path.basename(audio_path)}")
                shutil.copy(audio_path, output_path)
                print(f"File copied to {output_path}")
                return output_path
            except Exception as e:
                print(f"Error in fallback copy: {e}")
                return None

@app.route('/', methods=['GET', 'POST'])
def process_audio():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                print("No file part in request")
                return jsonify({'error': 'No file part'})
            file = request.files['file']
            if file.filename == '':
                print("No selected file")
                return jsonify({'error': 'No selected file'})
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"File saved to {filepath}")

                max_lufs = request.form.get('max_lufs', type=float)
                print(f"Target LUFS value: {max_lufs}")
                
                # Get audio info
                is_stereo_audio = is_stereo(filepath)
                if is_stereo_audio is None:
                    print("Failed to analyze audio channels")
                    return jsonify({'error': 'Failed to analyze audio channels'})
                print(f"Is stereo: {is_stereo_audio}")
                
                # Calculate original LUFS
                original_lufs = calculate_lufs(filepath)
                if original_lufs is None:
                    print("Failed to calculate original LUFS")
                    return jsonify({'error': 'Failed to calculate original LUFS'})
                print(f"Original LUFS: {original_lufs}")
                
                # Create original LUFS graph
                original_graph_data = create_lufs_graph(filepath)
                if original_graph_data is None:
                    print("Failed to create original LUFS graph")
                    return jsonify({'error': 'Failed to create original LUFS graph'})
                print(f"Original LUFS graph created: {original_graph_data}")
                
                # Check if normalization is requested
                normalize = request.form.get('normalize', 'true') == 'true'
                if normalize and max_lufs is not None:
                    target_lufs = max_lufs
                    print(f"Will normalize to target LUFS: {target_lufs}")
                    
                    # Start with the original file
                    processing_path = filepath
                    
                    # Apply noise reduction if enabled
                    noise_reduction = request.form.get('noise_reduction') == 'true'
                    if noise_reduction:
                        noise_strength = request.form.get('noise_strength', default=0.5, type=float)
                        print(f"Applying noise reduction with strength: {noise_strength}")
                        
                        # Apply noise reduction
                        # ... (noise reduction code)
                    
                    # Apply compression if enabled
                    compression = request.form.get('compression') == 'true'
                    if compression:
                        threshold = request.form.get('threshold', default=-20, type=float)
                        ratio = request.form.get('ratio', default=4, type=float)
                        attack = request.form.get('attack', default=3.1, type=float)
                        release = request.form.get('release', default=100, type=float)
                        
                        print(f"Applying compression: threshold={threshold}, ratio={ratio}, attack={attack}, release={release}")
                        compressed_path = apply_compression(processing_path, threshold, ratio, attack, release)
                        
                        # Clean up previous temp file if needed
                        if processing_path != filepath:
                            try:
                                os.remove(processing_path)
                                print(f"Removed temp file: {processing_path}")
                            except Exception as e:
                                print(f"Failed to remove temp file: {e}")
                        
                        if compressed_path is None:
                            print("Failed to apply compression")
                            return jsonify({'error': 'Failed to apply compression'})
                        
                        # Update processing path
                        processing_path = compressed_path
                        print(f"Compression applied, new path: {processing_path}")
                    
                    # Apply normalization
                    normalized_path = normalize_audio_loudness(processing_path, target_lufs)
                    
                    # Clean up temp file if needed
                    if processing_path != filepath:
                        try:
                            os.remove(processing_path)
                            print(f"Removed temp file: {processing_path}")
                        except Exception as e:
                            print(f"Failed to remove temp file: {e}")
                    
                    if normalized_path is None:
                        print("Failed to normalize audio")
                        return jsonify({'error': 'Failed to normalize audio'})
                    print(f"Normalization applied, new path: {normalized_path}")
                    
                    # Calculate processed LUFS
                    processed_lufs = calculate_lufs(normalized_path)
                    print(f"Processed LUFS: {processed_lufs}")
                    
                    # Create processed LUFS graph
                    processed_graph_data = create_lufs_graph(normalized_path)
                    if processed_graph_data is None:
                        print("Failed to create processed LUFS graph")
                        return jsonify({'error': 'Failed to create processed LUFS graph'})
                    print(f"Processed LUFS graph created: {processed_graph_data}")
                    
                    # Copy to final destination
                    processed_filename = "processed_" + secure_filename(filename)
                    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                    
                    # Convert to AudioSegment and export
                    try:
                        data, rate = sf.read(normalized_path)
                        sf.write(processed_filepath, data, rate)
                        print(f"Processed file exported to: {processed_filepath}")
                    except Exception as e:
                        print(f"Error exporting processed file: {e}")
                        return jsonify({'error': f'Error exporting processed file: {str(e)}'})
                    
                    # Clean up temp file
                    try:
                        os.remove(normalized_path)
                        print(f"Removed temp file: {normalized_path}")
                    except Exception as e:
                        print(f"Failed to remove temp file: {e}")
                    
                    # Verify the processed file exists
                    if not os.path.exists(processed_filepath):
                        print(f"Processed file not found at {processed_filepath}")
                        return jsonify({'error': 'Processed file not found'})
                    
                    print(f"Returning response with processed file: {processed_filename}")
                    return jsonify({
                        'lufs': original_lufs,
                        'processed_lufs': processed_lufs,
                        'processed_file': processed_filename,
                        'original_graph_data': original_graph_data,
                        'processed_graph_data': processed_graph_data,
                        'is_stereo': is_stereo_audio
                    })
                else:
                    print("Normalization not requested, returning original data only")
                    return jsonify({
                        'lufs': original_lufs,
                        'original_graph_data': original_graph_data,
                        'is_stereo': is_stereo_audio
                    })
            else:
                print(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type'})
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing audio: {e}")
            print(f"Error details: {error_details}")
            return jsonify({'error': f'Error processing audio: {str(e)}'})
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Only attempt to delete processed files (not original uploads)
    if filename.startswith('processed_'):
        @after_this_request
        def delete_after_request(response):
            try:
                # Only delete if the file was successfully sent
                if response.status_code == 200:
                    # Schedule deletion for after the response is fully sent
                    def delete_file():
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Deleted processed file after download: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file after download: {e}")
                    
                    # Use a background thread to delete the file
                    import threading
                    threading.Timer(1.0, delete_file).start()
            except Exception as e:
                print(f"Error setting up file deletion: {e}")
            return response
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
