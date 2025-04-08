document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const dataSourcesContainer = document.getElementById('data-sources-container');
    const loadingSources = document.getElementById('loading-sources');
    const saveConfigButton = document.getElementById('save-config');
    const startIngestionButton = document.getElementById('start-ingestion');
    const ingestionStatus = document.getElementById('ingestion-status');
    const statusMessage = document.getElementById('status-message');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    // Store data sources configuration
    let dataSourcesConfig = {};

    // Check if settings are configured
    checkSettings();

    // Load data sources configuration on page load
    loadDataSourcesConfig();

    // Add event listeners to buttons
    saveConfigButton.addEventListener('click', saveConfiguration);
    startIngestionButton.addEventListener('click', startIngestion);

    // Function to check if settings are configured
    async function checkSettings() {
        try {
            const response = await fetch('/api/settings/check');

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();

            if (!data.sources_configured) {
                // Show settings notification
                const notification = document.createElement('div');
                notification.className = 'settings-notification';
                notification.innerHTML = `
                    <div class="notification-content">
                        <h3>Data Sources Not Configured</h3>
                        <p>Please configure your data source settings before using the data ingestion feature.</p>
                        <a href="/ui/settings.html" class="button">Go to Settings</a>
                    </div>
                `;

                // Add notification to the page
                document.querySelector('main').prepend(notification);

                // Disable buttons
                saveConfigButton.disabled = true;
                startIngestionButton.disabled = true;
            }
        } catch (error) {
            console.error('Error checking settings:', error);
        }
    }

    // Function to load data sources configuration
    async function loadDataSourcesConfig() {
        try {
            // Show loading indicator
            loadingSources.classList.remove('hidden');

            // Fetch configuration from API
            const response = await fetch('/config/data_types');

            // Check if response is ok
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            // Parse response
            dataSourcesConfig = await response.json();

            // Render data sources
            renderDataSources();
        } catch (error) {
            console.error('Error loading data sources:', error);
            statusMessage.textContent = 'Error loading data sources configuration. Please try again.';
            ingestionStatus.classList.remove('hidden');
        } finally {
            // Hide loading indicator
            loadingSources.classList.add('hidden');
        }
    }

    // Function to render data sources
    function renderDataSources() {
        // Clear container
        dataSourcesContainer.innerHTML = '';

        // Create data source cards
        for (const [sourceId, sourceData] of Object.entries(dataSourcesConfig)) {
            const sourceCard = createDataSourceCard(sourceId, sourceData);
            dataSourcesContainer.appendChild(sourceCard);
        }
    }

    // Function to create a data source card
    function createDataSourceCard(sourceId, sourceData) {
        // Create card element
        const card = document.createElement('div');
        card.className = 'data-source-card';
        card.dataset.sourceId = sourceId;

        // Create header
        const header = document.createElement('div');
        header.className = 'data-source-header';

        // Create name element
        const name = document.createElement('div');
        name.className = 'data-source-name';
        name.textContent = sourceData.name || sourceId;

        // Create toggle switch
        const toggle = document.createElement('label');
        toggle.className = 'data-source-toggle';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = sourceData.enabled;
        checkbox.addEventListener('change', function() {
            dataSourcesConfig[sourceId].enabled = this.checked;
            updateDocumentTypesVisibility(sourceId, this.checked);
        });

        const slider = document.createElement('span');
        slider.className = 'toggle-slider';

        toggle.appendChild(checkbox);
        toggle.appendChild(slider);

        // Add elements to header
        header.appendChild(name);
        header.appendChild(toggle);

        // Create document types container
        const docTypesContainer = document.createElement('div');
        docTypesContainer.className = 'document-types';
        docTypesContainer.id = `doc-types-${sourceId}`;

        // Add document types
        if (sourceData.document_types) {
            for (const [typeId, typeData] of Object.entries(sourceData.document_types)) {
                const typeItem = createDocumentTypeItem(sourceId, typeId, typeData);
                docTypesContainer.appendChild(typeItem);
            }
        }

        // Update visibility based on enabled status
        updateDocumentTypesVisibility(sourceId, sourceData.enabled);

        // Add elements to card
        card.appendChild(header);
        card.appendChild(docTypesContainer);

        return card;
    }

    // Function to create a document type item
    function createDocumentTypeItem(sourceId, typeId, typeData) {
        // Create item element
        const item = document.createElement('div');
        item.className = 'document-type-item';

        // Create checkbox
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `${sourceId}-${typeId}`;
        checkbox.checked = typeData.enabled;
        checkbox.addEventListener('change', function() {
            dataSourcesConfig[sourceId].document_types[typeId].enabled = this.checked;
        });

        // Create label
        const label = document.createElement('label');
        label.htmlFor = `${sourceId}-${typeId}`;
        label.textContent = `${typeId} (${typeData.description || 'No description'})`;

        // Add elements to item
        item.appendChild(checkbox);
        item.appendChild(label);

        return item;
    }

    // Function to update document types visibility
    function updateDocumentTypesVisibility(sourceId, isEnabled) {
        const docTypesContainer = document.getElementById(`doc-types-${sourceId}`);
        if (docTypesContainer) {
            docTypesContainer.style.display = isEnabled ? 'block' : 'none';
        }
    }

    // Function to save configuration
    async function saveConfiguration() {
        try {
            // Show status
            ingestionStatus.classList.remove('hidden');
            statusMessage.textContent = 'Saving configuration...';

            // Prepare sources configuration
            const sourcesConfig = {};
            const typesConfig = {};

            for (const [sourceId, sourceData] of Object.entries(dataSourcesConfig)) {
                sourcesConfig[sourceId] = sourceData.enabled;

                if (sourceData.document_types) {
                    typesConfig[sourceId] = [];
                    for (const [typeId, typeData] of Object.entries(sourceData.document_types)) {
                        if (typeData.enabled) {
                            typesConfig[sourceId].push(typeId);
                        }
                    }
                }
            }

            // Send sources configuration to API
            const sourcesResponse = await fetch('/config/sources', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sources: sourcesConfig,
                    types: typesConfig
                })
            });

            // Check if response is ok
            if (!sourcesResponse.ok) {
                throw new Error(`API error: ${sourcesResponse.status}`);
            }

            // Update status
            statusMessage.textContent = 'Configuration saved successfully!';
        } catch (error) {
            console.error('Error saving configuration:', error);
            statusMessage.textContent = 'Error saving configuration. Please try again.';
        }
    }

    // Function to start ingestion
    async function startIngestion() {
        try {
            // Show status
            ingestionStatus.classList.remove('hidden');
            statusMessage.textContent = 'Starting data ingestion...';
            progressContainer.classList.remove('hidden');

            // Simulate progress (in a real implementation, this would be handled by the backend)
            let progress = 0;
            const interval = setInterval(() => {
                progress += 5;
                if (progress <= 100) {
                    updateProgress(progress);
                } else {
                    clearInterval(interval);
                    statusMessage.textContent = 'Data ingestion completed!';
                }
            }, 500);
        } catch (error) {
            console.error('Error starting ingestion:', error);
            statusMessage.textContent = 'Error starting data ingestion. Please try again.';
        }
    }

    // Function to update progress
    function updateProgress(percent) {
        progressFill.style.width = `${percent}%`;
        progressText.textContent = `${percent}%`;
    }
});