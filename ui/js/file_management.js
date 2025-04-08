document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const tabButtons = document.querySelectorAll('.tab-button');
    const currentCategoryTitle = document.getElementById('current-category');
    const uploadFileBtn = document.getElementById('upload-file-btn');
    const fileListContainer = document.getElementById('file-list-container');
    const loadingFiles = document.getElementById('loading-files');
    const fileList = document.getElementById('file-list');
    const emptyMessage = document.getElementById('empty-message');
    
    // Upload modal elements
    const uploadModal = document.getElementById('upload-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const cancelUploadBtn = document.getElementById('cancel-upload');
    const uploadForm = document.getElementById('upload-form');
    const categorySelect = document.getElementById('category-select');
    const fileInput = document.getElementById('file-input');
    const fileTypeInfo = document.getElementById('file-type-info');
    const uploadError = document.getElementById('upload-error');
    
    // Delete modal elements
    const deleteModal = document.getElementById('delete-modal');
    const closeDeleteModalBtn = document.getElementById('close-delete-modal');
    const cancelDeleteBtn = document.getElementById('cancel-delete');
    const confirmDeleteBtn = document.getElementById('confirm-delete');
    const deleteFilename = document.getElementById('delete-filename');
    
    // Store current category and file to delete
    let currentCategory = 'documents';
    let fileToDelete = null;
    
    // Store allowed file extensions for each category
    let allowedExtensions = {};
    
    // Load allowed extensions from data_types_config.json
    loadAllowedExtensions();
    
    // Load files for the default category
    loadFiles(currentCategory);
    
    // Add event listeners to tab buttons
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Update current category
            currentCategory = this.dataset.category;
            currentCategoryTitle.textContent = currentCategory.charAt(0).toUpperCase() + currentCategory.slice(1);
            
            // Load files for the selected category
            loadFiles(currentCategory);
        });
    });
    
    // Add event listener to upload button
    uploadFileBtn.addEventListener('click', function() {
        // Set category in modal
        categorySelect.value = currentCategory;
        
        // Update file type info
        updateFileTypeInfo(currentCategory);
        
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
    
    // Add event listener to close delete modal button
    closeDeleteModalBtn.addEventListener('click', function() {
        deleteModal.classList.add('hidden');
    });
    
    // Add event listener to cancel delete button
    cancelDeleteBtn.addEventListener('click', function() {
        deleteModal.classList.add('hidden');
    });
    
    // Add event listener to confirm delete button
    confirmDeleteBtn.addEventListener('click', function() {
        if (fileToDelete) {
            deleteFile(fileToDelete.category, fileToDelete.filename);
        }
    });
    
    // Function to load allowed extensions from data_types_config.json
    async function loadAllowedExtensions() {
        try {
            const response = await fetch('/api/file-extensions');
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            allowedExtensions = await response.json();
            
            // Update file type info for current category
            updateFileTypeInfo(currentCategory);
            
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
    
    // Function to load files for a category
    async function loadFiles(category) {
        try {
            // Show loading indicator
            loadingFiles.classList.remove('hidden');
            fileList.classList.add('hidden');
            emptyMessage.classList.add('hidden');
            
            // Fetch files from API
            const response = await fetch(`/api/files/${category}`);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const files = await response.json();
            
            // Clear file list
            fileList.innerHTML = '';
            
            // Check if there are files
            if (files.length === 0) {
                // Show empty message
                loadingFiles.classList.add('hidden');
                emptyMessage.classList.remove('hidden');
                return;
            }
            
            // Add files to list
            files.forEach(file => {
                const fileCard = document.createElement('div');
                fileCard.className = 'file-card';
                
                const fileName = document.createElement('div');
                fileName.className = 'file-name';
                fileName.textContent = file.name;
                
                const fileInfo = document.createElement('div');
                fileInfo.className = 'file-info';
                fileInfo.textContent = `Size: ${formatFileSize(file.size)} | Type: ${file.type}`;
                
                const fileActions = document.createElement('div');
                fileActions.className = 'file-actions';
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'file-action-btn delete-btn';
                deleteBtn.textContent = 'Delete';
                deleteBtn.addEventListener('click', function() {
                    // Set file to delete
                    fileToDelete = {
                        category: category,
                        filename: file.name
                    };
                    
                    // Set filename in modal
                    deleteFilename.textContent = file.name;
                    
                    // Show delete modal
                    deleteModal.classList.remove('hidden');
                });
                
                fileActions.appendChild(deleteBtn);
                
                fileCard.appendChild(fileName);
                fileCard.appendChild(fileInfo);
                fileCard.appendChild(fileActions);
                
                fileList.appendChild(fileCard);
            });
            
            // Hide loading indicator and show file list
            loadingFiles.classList.add('hidden');
            fileList.classList.remove('hidden');
            
        } catch (error) {
            console.error(`Error loading ${category} files:`, error);
            
            // Hide loading indicator and show error message
            loadingFiles.classList.add('hidden');
            emptyMessage.textContent = `Error loading files: ${error.message}`;
            emptyMessage.classList.remove('hidden');
        }
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
            
            // Reload files for the current category
            loadFiles(currentCategory);
            
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
    
    // Function to delete a file
    async function deleteFile(category, filename) {
        try {
            // Disable delete button
            confirmDeleteBtn.disabled = true;
            
            // Delete file from API
            const response = await fetch(`/api/files/${category}/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            
            // Check if delete was successful
            if (!response.ok) {
                const result = await response.json();
                throw new Error(result.detail || 'Delete failed');
            }
            
            // Hide delete modal
            deleteModal.classList.add('hidden');
            
            // Reload files for the current category
            loadFiles(currentCategory);
            
        } catch (error) {
            console.error('Error deleting file:', error);
            alert(`Error deleting file: ${error.message}`);
        } finally {
            // Enable delete button
            confirmDeleteBtn.disabled = false;
        }
    }
    
    // Function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});
