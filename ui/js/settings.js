document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const setupWizard = document.getElementById('setup-wizard');
    const startWizardButton = document.getElementById('start-wizard');
    const nextToLlmButton = document.getElementById('next-to-llm');
    const backToSourcesButton = document.getElementById('back-to-sources');
    const nextToCredentialsButton = document.getElementById('next-to-credentials');
    const backToLlmButton = document.getElementById('back-to-llm');
    const saveSettingsButton = document.getElementById('save-settings');
    const settingsStatus = document.getElementById('settings-status');
    const statusMessage = document.getElementById('status-message');

    // Get wizard steps
    const stepWelcome = document.getElementById('step-welcome');
    const stepDataSources = document.getElementById('step-data-sources');
    const stepLlm = document.getElementById('step-llm');
    const stepCredentials = document.getElementById('step-credentials');
    const stepSuccess = document.getElementById('step-success');

    // Get data source checkboxes
    const githubSourceCheckbox = document.getElementById('github-source');
    const onenoteSourceCheckbox = document.getElementById('onenote-source');

    // Get LLM radio buttons
    const gptModelRadio = document.getElementById('gpt-model');
    const claudeModelRadio = document.getElementById('claude-model');
    const llamaModelRadio = document.getElementById('llama-model');

    // Get credential groups
    const gptCredentials = document.getElementById('gpt-credentials');
    const claudeCredentials = document.getElementById('claude-credentials');
    const githubCredentials = document.getElementById('github-credentials');
    const onenoteCredentials = document.getElementById('onenote-credentials');

    // Load current settings on page load
    loadCurrentSettings();

    // Add event listeners to wizard buttons
    startWizardButton.addEventListener('click', function() {
        showStep(stepDataSources);
    });

    nextToLlmButton.addEventListener('click', function() {
        showStep(stepLlm);
    });

    backToSourcesButton.addEventListener('click', function() {
        showStep(stepDataSources);
    });

    nextToCredentialsButton.addEventListener('click', function() {
        showStep(stepCredentials);
        updateCredentialGroups();
    });

    backToLlmButton.addEventListener('click', function() {
        showStep(stepLlm);
    });

    saveSettingsButton.addEventListener('click', saveSettings);

    // Add event listeners to data source checkboxes
    githubSourceCheckbox.addEventListener('change', updateCredentialGroups);
    onenoteSourceCheckbox.addEventListener('change', updateCredentialGroups);

    // Add event listeners to LLM radio buttons
    gptModelRadio.addEventListener('change', updateCredentialGroups);
    claudeModelRadio.addEventListener('change', updateCredentialGroups);
    llamaModelRadio.addEventListener('change', updateCredentialGroups);

    // Function to show a specific step and hide others
    function showStep(stepToShow) {
        // Hide all steps
        stepWelcome.classList.add('hidden');
        stepDataSources.classList.add('hidden');
        stepLlm.classList.add('hidden');
        stepCredentials.classList.add('hidden');
        stepSuccess.classList.add('hidden');

        // Show the specified step
        stepToShow.classList.remove('hidden');
    }

    // Function to update credential groups based on selections
    function updateCredentialGroups() {
        // Update LLM credential groups
        gptCredentials.classList.add('hidden');
        claudeCredentials.classList.add('hidden');

        if (gptModelRadio.checked) {
            gptCredentials.classList.remove('hidden');
        } else if (claudeModelRadio.checked) {
            claudeCredentials.classList.remove('hidden');
        }

        // Update data source credential groups
        githubCredentials.classList.toggle('hidden', !githubSourceCheckbox.checked);
        onenoteCredentials.classList.toggle('hidden', !onenoteSourceCheckbox.checked);
    }

    // Function to load current settings
    async function loadCurrentSettings() {
        try {
            // Fetch current settings from API
            const response = await fetch('/api/settings');

            // Check if response is ok
            if (!response.ok) {
                console.log('No existing settings found or error fetching settings');
                return;
            }

            // Parse response
            const settings = await response.json();

            // Update form fields with current settings
            if (settings.openai_api_key) {
                document.getElementById('openai-api-key').value = settings.openai_api_key;
                gptModelRadio.checked = true;
            }

            if (settings.anthropic_api_key) {
                document.getElementById('anthropic-api-key').value = settings.anthropic_api_key;
                if (!settings.openai_api_key) {
                    claudeModelRadio.checked = true;
                }
            }

            if (settings.github_token) {
                document.getElementById('github-token').value = settings.github_token;
                githubSourceCheckbox.checked = true;
            }

            if (settings.microsoft_client_id) {
                document.getElementById('microsoft-client-id').value = settings.microsoft_client_id;
                document.getElementById('microsoft-client-secret').value = settings.microsoft_client_secret || '';
                document.getElementById('microsoft-tenant-id').value = settings.microsoft_tenant_id || '';
                onenoteSourceCheckbox.checked = true;
            }

            if (settings.data_type_config) {
                document.getElementById('data-type-config').value = settings.data_type_config;
            }

            // Update credential groups
            updateCredentialGroups();

        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    // Function to save settings
    async function saveSettings() {
        try {
            // Show status
            settingsStatus.classList.remove('hidden');
            statusMessage.textContent = 'Saving settings...';

            // Get form values
            const settings = {
                openai_api_key: gptModelRadio.checked ? document.getElementById('openai-api-key').value : '',
                anthropic_api_key: claudeModelRadio.checked ? document.getElementById('anthropic-api-key').value : '',
                github_token: githubSourceCheckbox.checked ? document.getElementById('github-token').value : '',
                microsoft_client_id: onenoteSourceCheckbox.checked ? document.getElementById('microsoft-client-id').value : '',
                microsoft_client_secret: onenoteSourceCheckbox.checked ? document.getElementById('microsoft-client-secret').value : '',
                microsoft_tenant_id: onenoteSourceCheckbox.checked ? document.getElementById('microsoft-tenant-id').value : '',
                data_type_config: document.getElementById('data-type-config').value,
                llm_type: gptModelRadio.checked ? 'gpt' : (claudeModelRadio.checked ? 'claude' : 'llama4'),
                data_sources: {
                    local: document.getElementById('local-source').checked,
                    github: githubSourceCheckbox.checked,
                    onenote: onenoteSourceCheckbox.checked
                }
            };

            // Validate required fields
            let validationErrors = [];

            if (gptModelRadio.checked && !settings.openai_api_key) {
                validationErrors.push('OpenAI API Key is required for GPT model');
            }

            if (claudeModelRadio.checked && !settings.anthropic_api_key) {
                validationErrors.push('Anthropic API Key is required for Claude model');
            }

            if (githubSourceCheckbox.checked && !settings.github_token) {
                validationErrors.push('GitHub Token is required for GitHub data source');
            }

            if (onenoteSourceCheckbox.checked) {
                if (!settings.microsoft_client_id) {
                    validationErrors.push('Microsoft Client ID is required for OneNote data source');
                }
                if (!settings.microsoft_client_secret) {
                    validationErrors.push('Microsoft Client Secret is required for OneNote data source');
                }
                if (!settings.microsoft_tenant_id) {
                    validationErrors.push('Microsoft Tenant ID is required for OneNote data source');
                }
            }

            if (!settings.data_type_config) {
                validationErrors.push('Data Types Configuration File is required');
            }

            // Display validation errors if any
            if (validationErrors.length > 0) {
                statusMessage.innerHTML = '<strong>Validation Errors:</strong><ul>' +
                    validationErrors.map(error => `<li>${error}</li>`).join('') + '</ul>';
                return;
            }

            // Send settings to API
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            // Check if response is ok
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            // The backend now handles updating data_types_config.json
            // No need to make a separate API call

            // Show success message
            statusMessage.textContent = 'Settings saved successfully!';

            // Show success step
            showStep(stepSuccess);

        } catch (error) {
            console.error('Error saving settings:', error);
            statusMessage.textContent = 'Error saving settings: ' + error.message;
        }
    }

    // The backend now handles updating data_types_config.json directly
    // No need for a separate function
});
