document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const llmSelect = document.getElementById('llm-select');
    const queryMethodSelect = document.getElementById('query-method-select');
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-query');
    const loadingIndicator = document.getElementById('loading-indicator');
    const queryResults = document.getElementById('query-results');
    const answerText = document.getElementById('answer-text');
    const sourcesList = document.getElementById('sources-list');
    const quickUploadBtn = document.getElementById('quick-upload-btn');

    // Upload modal elements
    const uploadModal = document.getElementById('upload-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const cancelUploadBtn = document.getElementById('cancel-upload');
    const uploadForm = document.getElementById('upload-form');
    const categorySelect = document.getElementById('category-select');
    const fileInput = document.getElementById('file-input');
    const fileTypeInfo = document.getElementById('file-type-info');
    const uploadError = document.getElementById('upload-error');

    // Store allowed file extensions for each category
    let allowedExtensions = {};

    // Check if settings are configured
    checkSettings();

    // Load allowed extensions from data_types_config.json
    loadAllowedExtensions();

    // Add event listener to submit button
    submitButton.addEventListener('click', handleQuerySubmit);

    // Add event listener to quick upload button
    quickUploadBtn.addEventListener('click', function() {
        // Clear previous file selection and error
        fileInput.value = '';
        uploadError.classList.add('hidden');

        // Show modal
        uploadModal.classList.remove('hidden');
    });

    // Add event listener to close modal button
    closeModalBtn.addEventListener('click', function() {
        uploadModal.classList.add('hidden');
    });

    // Add event listener to cancel upload button
    cancelUploadBtn.addEventListener('click', function() {
        uploadModal.classList.add('hidden');
    });

    // Add event listener to category select
    categorySelect.addEventListener('change', function() {
        updateFileTypeInfo(this.value);
    });

    // Add event listener to upload form
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Get selected file
        const file = fileInput.files[0];

        // Validate file
        if (!file) {
            showUploadError('Please select a file to upload.');
            return;
        }

        // Get selected category
        const category = categorySelect.value;

        // Validate file extension
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions[category] || !allowedExtensions[category].includes(fileExtension)) {
            showUploadError(`Invalid file type. Allowed extensions for ${category}: ${allowedExtensions[category].join(', ')}`);
            return;
        }

        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', category);

        // Upload file
        uploadFile(formData);
    });

    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);

    // Initial resize
    autoResizeTextarea.call(queryInput);

    // Function to auto-resize textarea
    function autoResizeTextarea() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    }

    // Function to load allowed extensions from data_types_config.json
    async function loadAllowedExtensions() {
        try {
            const response = await fetch('/api/file-extensions');

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            allowedExtensions = await response.json();

            // Update file type info for current category
            updateFileTypeInfo(categorySelect.value);

        } catch (error) {
            console.error('Error loading allowed extensions:', error);
        }
    }

    // Function to update file type info
    function updateFileTypeInfo(category) {
        if (allowedExtensions[category]) {
            fileTypeInfo.textContent = `Allowed extensions: ${allowedExtensions[category].join(', ')}`;
        } else {
            fileTypeInfo.textContent = 'Loading allowed extensions...';
        }
    }

    // Function to show upload error
    function showUploadError(message) {
        uploadError.textContent = message;
        uploadError.classList.remove('hidden');
    }

    // Function to upload a file
    async function uploadFile(formData) {
        try {
            // Disable form elements
            categorySelect.disabled = true;
            fileInput.disabled = true;
            document.getElementById('confirm-upload').disabled = true;
            document.getElementById('cancel-upload').disabled = true;

            // Show loading message
            uploadError.textContent = 'Uploading file...';
            uploadError.classList.remove('hidden');

            // Upload file to API
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            // Parse response
            const result = await response.json();

            // Check if upload was successful
            if (!response.ok) {
                throw new Error(result.detail || 'Upload failed');
            }

            // Hide modal
            uploadModal.classList.add('hidden');

            // Show success message
            alert(`File uploaded successfully to ${formData.get('category')}!`);

        } catch (error) {
            console.error('Error uploading file:', error);
            showUploadError(`Error uploading file: ${error.message}`);
        } finally {
            // Enable form elements
            categorySelect.disabled = false;
            fileInput.disabled = false;
            document.getElementById('confirm-upload').disabled = false;
            document.getElementById('cancel-upload').disabled = false;
        }
    }

    // Function to check if settings are configured
    async function checkSettings() {
        try {
            const response = await fetch('/api/settings/check');

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();

            if (!data.llm_configured) {
                // Show settings notification
                const notification = document.createElement('div');
                notification.className = 'settings-notification';
                notification.innerHTML = `
                    <div class="notification-content">
                        <h3>LLM Not Configured</h3>
                        <p>Please configure your LLM settings before using the query feature.</p>
                        <a href="/ui/settings.html" class="button">Go to Settings</a>
                    </div>
                `;

                // Add notification to the page
                document.querySelector('main').prepend(notification);

                // Disable submit button
                submitButton.disabled = true;
            }
        } catch (error) {
            console.error('Error checking settings:', error);
        }
    }

    // Function to handle query submission
    async function handleQuerySubmit() {
        // Get values from form
        const llm = llmSelect.value;
        const queryMethod = queryMethodSelect.value;
        const question = queryInput.value.trim();

        // Validate input
        if (!question) {
            alert('Please enter a question');
            return;
        }

        // Show loading indicator and hide results
        loadingIndicator.classList.remove('hidden');
        queryResults.classList.add('hidden');

        try {
            // Prepare request data
            const requestData = {
                question: question,
                k: 10,
                llm: llm,
                query_method: queryMethod
            };

            // Send request to API
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            // Check if response is ok
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            // Parse response
            const data = await response.json();

            // Display results
            displayResults(data);
        } catch (error) {
            console.error('Error submitting query:', error);
            alert('Error submitting query. Please try again.');
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
        }
    }

    // Function to display query results
    function displayResults(data) {
        // Display answer
        answerText.textContent = data.answer;

        // Clear previous sources
        sourcesList.innerHTML = '';

        // Add sources to list
        data.sources.forEach(source => {
            const listItem = document.createElement('li');
            listItem.textContent = source;
            sourcesList.appendChild(listItem);
        });

        // Show results
        queryResults.classList.remove('hidden');
    }
});