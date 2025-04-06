// File input handling
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Get all the elements - FIXED FILE INPUT ID
    const fileInput = document.getElementById('file'); // Changed from 'file-input' to 'file'
    const fileNameDisplay = document.getElementById('file-name');
    const uploadButton = document.getElementById('upload-button');
    const uploadForm = document.getElementById('upload-form');
    const uploadError = document.getElementById('upload-error');
    const spinner = document.getElementById('spinner');
    
    // Debug file input element
    console.log("File input element:", fileInput);
    
    // Compression elements
    const compressionCheckbox = document.getElementById('compression');
    const compressionOptions = document.getElementById('compression-options');
    const compressionInfoBtn = document.getElementById('compression-info-btn');
    const compressionInfo = document.getElementById('compression-info');
    
    // Noise reduction elements
    const noiseReductionCheckbox = document.getElementById('noise_reduction');
    const noiseReductionOptions = document.getElementById('noise_reduction_options');
    const noiseReductionInfoBtn = document.getElementById('noise_reduction_info_btn');
    const noiseReductionInfo = document.getElementById('noise_reduction_info');
    const noiseStrengthSlider = document.getElementById('noise_strength');
    const noiseStrengthValue = document.getElementById('noise_strength_value');
    
    // Debug element existence
    console.log("File input:", fileInput);
    console.log("File name display:", fileNameDisplay);
    console.log("Upload form:", uploadForm);
    console.log("Compression checkbox:", compressionCheckbox);
    console.log("Compression info button:", compressionInfoBtn);
    console.log("Compression info div:", compressionInfo);
    console.log("Noise reduction info button:", noiseReductionInfoBtn);
    console.log("Noise reduction info div:", noiseReductionInfo);
    
    // Handle file input change
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileNameDisplay.textContent = this.files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file selected';
            }
        });
    }
    
    // Handle compression checkbox
    if (compressionCheckbox && compressionOptions) {
        compressionCheckbox.addEventListener('change', function() {
            console.log("Compression checkbox changed:", this.checked);
            compressionOptions.style.display = this.checked ? 'block' : 'none';
        });
    }
    
    // Handle compression info button
    if (compressionInfoBtn && compressionInfo) {
        compressionInfoBtn.addEventListener('click', function(e) {
            e.preventDefault(); // Prevent any default button behavior
            console.log("Compression info button clicked");
            compressionInfo.classList.toggle('hidden');
        });
    }
    
    // Handle noise reduction checkbox
    if (noiseReductionCheckbox && noiseReductionOptions) {
        noiseReductionCheckbox.addEventListener('change', function() {
            console.log("Noise reduction checkbox changed:", this.checked);
            noiseReductionOptions.style.display = this.checked ? 'block' : 'none';
        });
    }
    
    // Handle noise reduction info button
    if (noiseReductionInfoBtn && noiseReductionInfo) {
        noiseReductionInfoBtn.addEventListener('click', function(e) {
            e.preventDefault(); // Prevent any default button behavior
            console.log("Noise reduction info button clicked");
            noiseReductionInfo.classList.toggle('hidden');
        });
    }
    
    // Handle noise strength slider
    if (noiseStrengthSlider && noiseStrengthValue) {
        noiseStrengthSlider.addEventListener('input', function() {
            noiseStrengthValue.textContent = this.value;
        });
    }

    // Form submission handling
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            console.log("Form submitted");

            // Get file input element directly
            const fileInput = document.getElementById('file');
            
            if (!fileInput || !fileInput.files || !fileInput.files[0]) {
                console.error("No file selected");
                alert('Please select an audio file.');
                return;
            }

            const file = fileInput.files[0];
            console.log("Selected file:", file);
            
            const maxLufs = document.getElementById('target_lufs')?.value;
            console.log("Target LUFS:", maxLufs);

            if (!maxLufs) {
                console.error("No LUFS value entered");
                alert('Please enter a target LUFS value.');
                return;
            }

            // Disable button and show spinner
            if (uploadButton) {
                uploadButton.disabled = true;
                document.getElementById('button-text').textContent = 'Processing...';
            }
            if (spinner) spinner.style.display = 'inline-flex';
            
            // Clear previous graphs
            clearGraphs();
            
            // Clear previous error messages
            if (uploadError) uploadError.textContent = '';

            const formData = new FormData();
            formData.append('file', file);
            formData.append('max_lufs', maxLufs);
            formData.append('normalize', 'true'); // Ensure normalization is requested
            
            // Add compression parameters if enabled
            if (compressionCheckbox && compressionCheckbox.checked) {
                formData.append('compression', 'true');
                formData.append('threshold', document.getElementById('threshold')?.value || '-20');
                formData.append('ratio', document.getElementById('ratio')?.value || '4');
                formData.append('attack', document.getElementById('attack')?.value || '3.1');
                formData.append('release', document.getElementById('release')?.value || '100');
            }
            
            // Add noise reduction parameters if enabled
            if (noiseReductionCheckbox && noiseReductionCheckbox.checked) {
                formData.append('noise_reduction', 'true');
                formData.append('noise_strength', document.getElementById('noise_strength')?.value || '0.5');
            }

            // Use absolute URL to ensure consistency
            const endpoint = window.location.origin + '/';
            console.log("Sending request to:", endpoint);

            fetch(endpoint, {
                method: 'POST',
                body: formData,
                cache: 'no-store' // Prevent caching issues
            })
            .then(response => {
                console.log("Response status:", response.status);
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Response data:", data);
                
                if (data.error) {
                    console.error("Server error:", data.error);
                    if (uploadError) uploadError.textContent = data.error;
                    
                    // Re-enable button even on error
                    if (uploadButton) {
                        uploadButton.disabled = false;
                        document.getElementById('button-text').textContent = 'Analyze Audio';
                    }
                    if (spinner) spinner.style.display = 'none';
                    return;
                }
                
                // Re-enable button
                if (uploadButton) {
                    uploadButton.disabled = false;
                    document.getElementById('button-text').textContent = 'Analyze Audio';
                }
                if (spinner) spinner.style.display = 'none';
                
                // Display results
                const resultsDiv = document.getElementById('results');
                if (resultsDiv) {
                    resultsDiv.style.display = 'block';
                    
                    // Update LUFS values
                    const lufsValue = document.getElementById('lufs-value');
                    if (lufsValue) lufsValue.textContent = data.lufs ? data.lufs.toFixed(1) + ' LUFS' : 'N/A';
                    
                    const processedLufsValue = document.getElementById('processed-lufs-value');
                    if (processedLufsValue) {
                        if (data.processed_lufs) {
                            processedLufsValue.textContent = data.processed_lufs.toFixed(1) + ' LUFS';
                            document.getElementById('processed-lufs-container').style.display = 'block';
                        } else {
                            document.getElementById('processed-lufs-container').style.display = 'none';
                        }
                    }
                    
                    // Show download link if processed file exists
                    const downloadLink = document.getElementById('download-link');
                    if (downloadLink) {
                        if (data.processed_file) {
                            downloadLink.href = '/uploads/' + data.processed_file;
                            downloadLink.style.display = 'inline-block';
                            console.log("Download link set to:", downloadLink.href);
                        } else {
                            downloadLink.style.display = 'none';
                            console.log("No processed file available, hiding download link");
                        }
                    }
                    
                    // Render graphs if available
                    console.log("Original graph data:", data.original_graph_data);
                    console.log("Processed graph data:", data.processed_graph_data);
                    
                    if (data.original_graph_data) {
                        const originalCanvas = document.getElementById('original-graph');
                        console.log("Original graph canvas:", originalCanvas);
                        if (originalCanvas) {
                            renderGraph(
                                originalCanvas, 
                                data.original_graph_data, 
                                'Original Audio LUFS'
                            );
                        } else {
                            console.error("Original graph canvas not found");
                        }
                    } else {
                        console.error("No original graph data received");
                    }
                    
                    if (data.processed_graph_data) {
                        const processedCanvas = document.getElementById('processed-graph');
                        console.log("Processed graph canvas:", processedCanvas);
                        if (processedCanvas) {
                            renderGraph(
                                processedCanvas, 
                                data.processed_graph_data, 
                                'Processed Audio LUFS'
                            );
                            document.getElementById('processed-graph-container').style.display = 'block';
                        } else {
                            console.error("Processed graph canvas not found");
                        }
                    } else {
                        document.getElementById('processed-graph-container').style.display = 'none';
                    }
                }
            })
            .catch(error => {
                console.error("Fetch error:", error);
                if (uploadError) uploadError.textContent = 'An unexpected error occurred: ' + error.message;
                if (uploadButton) {
                    uploadButton.disabled = false;
                    document.getElementById('button-text').textContent = 'Analyze Audio';
                }
                if (spinner) spinner.style.display = 'none';
            });
        });
    }

    // Function to clear existing graphs
    function clearGraphs() {
        const originalCanvas = document.getElementById('original-graph');
        const processedCanvas = document.getElementById('processed-graph');
        
        if (originalCanvas && originalCanvas.chart) {
            originalCanvas.chart.destroy();
        }
        
        if (processedCanvas && processedCanvas.chart) {
            processedCanvas.chart.destroy();
        }
    }
    
    // Graph rendering function
    function renderGraph(canvasElement, data, title) {
        if (!canvasElement || !data || !data.time || !data.loudness) {
            console.error('Missing data for graph rendering', {
                canvasElement: !!canvasElement,
                data: !!data,
                time: data?.time ? 'present' : 'missing',
                loudness: data?.loudness ? 'present' : 'missing'
            });
            return;
        }
        
        const ctx = canvasElement.getContext('2d');
        if (!ctx) {
            console.error('Failed to get canvas context');
            return;
        }

        // Destroy existing chart if it exists
        if (canvasElement.chart) {
            canvasElement.chart.destroy();
        }
        
        console.log(`Rendering ${title} graph with ${data.time.length} data points`);
        console.log("Time values:", data.time.slice(0, 5), "...");
        console.log("Loudness values:", data.loudness.slice(0, 5), "...");

        // Create new chart
        canvasElement.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.time.map(t => t.toFixed(1) + 's'),
                datasets: [{
                    label: 'LUFS',
                    data: data.loudness,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.raw.toFixed(1) + ' LUFS';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: Math.min(-70, data.min - 5),
                        max: Math.max(-10, data.max + 5),
                        title: {
                            display: true,
                            text: 'LUFS'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
    }
});
