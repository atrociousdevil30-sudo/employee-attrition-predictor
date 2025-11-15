document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('attritionForm');
    const datasetForm = document.getElementById('datasetForm');
    const resultCard = document.getElementById('resultCard');
    const loadingElement = document.getElementById('loadingOverlay');
    const riskScoreElement = document.getElementById('riskScore');
    const riskLevelElement = document.getElementById('riskLevel');
    const riskDescriptionElement = document.getElementById('riskDescription');
    const riskDriversElement = document.getElementById('riskDrivers');
    const datasetStatusElement = document.getElementById('datasetStatus');
    const employeeSelect = document.getElementById('employeeSelect');
    const employeeSelectGroup = document.getElementById('employeeSelectGroup');

    let uploadedEmployees = [];
    let previousPrediction = null;

    const jobRoleSelect = document.getElementById('jobRole');
    async function fetchAndPopulateJobRoles() {
        try {
            const response = await fetch('/api/job_roles');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            const roles = Array.isArray(data.job_roles) ? data.job_roles : [];
            jobRoleSelect.innerHTML = '';

            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = '-- Select Job Role --';
            defaultOption.disabled = true;
            defaultOption.selected = true;
            jobRoleSelect.appendChild(defaultOption);

            roles.forEach(role => {
                const option = document.createElement('option');
                option.value = role;
                option.textContent = role;
                jobRoleSelect.appendChild(option);
            });
        } catch (err) {
            const fallback = [
                'Healthcare Representative',
                'Research Scientist',
                'Sales Executive',
                'Laboratory Technician',
                'Manufacturing Director',
                'Research Director',
                'Human Resources',
                'Sales Representative',
                'Manager'
            ];

            jobRoleSelect.innerHTML = '';
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = '-- Select Job Role --';
            defaultOption.disabled = true;
            defaultOption.selected = true;
            jobRoleSelect.appendChild(defaultOption);

            fallback.forEach(role => {
                const option = document.createElement('option');
                option.value = role;
                option.textContent = role;
                jobRoleSelect.appendChild(option);
            });
        }
    }

    fetchAndPopulateJobRoles();

    if (datasetForm) {
        datasetForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('datasetFile');
            if (!fileInput.files || fileInput.files.length === 0) {
                datasetStatusElement.textContent = 'Please select a CSV file to upload.';
                datasetStatusElement.style.color = 'var(--danger-color)';
                return;
            }

            const formData = new FormData();
            formData.append('dataset', fileInput.files[0]);

            datasetStatusElement.textContent = 'Uploading dataset...';
            datasetStatusElement.style.color = '';

            try {
                const response = await fetch('/api/upload_dataset', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Upload failed');
                }

                datasetStatusElement.textContent = data.message || 'Dataset uploaded successfully.';
                datasetStatusElement.style.color = 'var(--primary-color)';

                // If backend returned a sample employee, auto-fill the form
                if (data.sample_employee) {
                    const sample = data.sample_employee;
                    const form = document.getElementById('attritionForm');
                    if (form) {
                        if (typeof sample.Age !== 'undefined') {
                            form.age.value = sample.Age;
                        }
                        if (typeof sample.JobRole !== 'undefined') {
                            form.jobRole.value = sample.JobRole;
                        }
                        if (typeof sample.MonthlyIncome !== 'undefined') {
                            form.monthlyIncome.value = sample.MonthlyIncome;
                        }
                        if (typeof sample.YearsAtCompany !== 'undefined') {
                            form.yearsAtCompany.value = sample.YearsAtCompany;
                        }

                        if (typeof sample.Gender === 'string') {
                            const genderValue = sample.Gender.toLowerCase() === 'female' ? 'female' : 'male';
                            const genderInput = document.querySelector(`input[name="gender"][value="${genderValue}"]`);
                            if (genderInput) {
                                genderInput.checked = true;
                            }
                        }

                        if (typeof sample.OverTime === 'string') {
                            form.overTime.checked = sample.OverTime.toLowerCase() === 'yes';
                        }
                    }
                }

                // If backend returned employees list, populate the selector
                if (Array.isArray(data.employees) && employeeSelect) {
                    uploadedEmployees = data.employees;
                    // Reset options
                    employeeSelect.innerHTML = '<option value="">-- Choose an employee --</option>';

                    uploadedEmployees.forEach((emp, index) => {
                        const opt = document.createElement('option');
                        const empNumber = emp.EmployeeNumber ?? index + 1;
                        const labelParts = [
                            `#${empNumber}`,
                            emp.JobRole ? String(emp.JobRole) : '',
                            typeof emp.Age !== 'undefined' ? `Age ${emp.Age}` : ''
                        ].filter(Boolean);
                        opt.value = String(index);
                        opt.textContent = labelParts.join(' - ');
                        employeeSelect.appendChild(opt);
                    });

                    if (employeeSelectGroup) {
                        employeeSelectGroup.style.display = 'block';
                    }
                }
            } catch (err) {
                datasetStatusElement.textContent = `Error uploading dataset: ${err.message}`;
                datasetStatusElement.style.color = 'var(--danger-color)';
            }
        });
    }

    if (employeeSelect) {
        employeeSelect.addEventListener('change', () => {
            const index = parseInt(employeeSelect.value, 10);
            if (Number.isNaN(index) || index < 0 || index >= uploadedEmployees.length) {
                return;
            }

            const emp = uploadedEmployees[index];
            const form = document.getElementById('attritionForm');
            if (!form) return;

            if (typeof emp.Age !== 'undefined') {
                form.age.value = emp.Age;
            }
            if (typeof emp.JobRole !== 'undefined') {
                form.jobRole.value = emp.JobRole;
            }
            if (typeof emp.MonthlyIncome !== 'undefined') {
                form.monthlyIncome.value = emp.MonthlyIncome;
            }
            if (typeof emp.YearsAtCompany !== 'undefined') {
                form.yearsAtCompany.value = emp.YearsAtCompany;
            }

            if (typeof emp.Gender === 'string') {
                const genderValue = emp.Gender.toLowerCase() === 'female' ? 'female' : 'male';
                const genderInput = document.querySelector(`input[name="gender"][value="${genderValue}"]`);
                if (genderInput) {
                    genderInput.checked = true;
                }
            }

            if (typeof emp.OverTime === 'string') {
                form.overTime.checked = emp.OverTime.toLowerCase() === 'yes';
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        loadingElement.style.display = 'block';
        resultCard.style.display = 'none';
        
        try {
            // Get form data with proper type conversion and formatting
            const formData = {
                age: parseInt(form.age.value, 10),
                jobRole: form.jobRole.value,
                monthlyIncome: parseFloat(form.monthlyIncome.value),
                yearsAtCompany: parseFloat(form.yearsAtCompany.value),
                overTime: form.overTime.checked ? 'Yes' : 'No',  // Convert boolean to 'Yes'/'No' string
                gender: document.querySelector('input[name="gender"]:checked')?.value || 'Male'  // Ensure proper capitalization
            };
            
            // Basic validation
            if (isNaN(formData.age) || isNaN(formData.monthlyIncome) || isNaN(formData.yearsAtCompany)) {
                throw new Error('Please enter valid numbers for all required fields');
            }
            
            if (!formData.jobRole) {
                throw new Error('Please select a job role');
            }
            
            // Call the prediction API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update UI with results
            displayResults(data);
            await findSimilar(formData);
            
        } catch (error) {
            console.error('Error:', error);
            riskDescriptionElement.textContent = `An error occurred: ${error.message || 'Please try again later.'}`;
            riskDescriptionElement.style.color = 'var(--danger-color)';
            resultCard.style.display = 'block';
        } finally {
            loadingElement.style.display = 'none';
        }
    });
    
    function displayResults(data) {
        const riskScore = data.risk;
        const riskLevel = data.risk_level || 'Medium';
        const factors = data.factors || [];
        const retentionSuggestions = data.retention_suggestions || [];
        
        // Add definition of Attrition
        const attritionDefinition = document.createElement('div');
        attritionDefinition.className = 'attrition-definition';
        attritionDefinition.innerHTML = `
            <div style="background: rgba(0, 188, 212, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid var(--primary-color);">
                <p style="margin: 0 0 8px 0; font-weight: 600; color: var(--primary-color);">About Employee Attrition:</p>
                <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #e0e0e0;">
                    Attrition refers to the gradual reduction in workforce as employees leave the company and are not replaced. 
                    It's important to monitor and understand attrition risks to maintain a stable and productive work environment.
                </p>
            </div>
        `;
        
        // Insert the definition at the top of the results
        if (riskDriversElement.firstChild) {
            riskDriversElement.insertBefore(attritionDefinition, riskDriversElement.firstChild);
        }
        
        // Update risk score with animation
        animateValue(riskScoreElement, 0, riskScore, 1000);
        
        // Set color and description based on risk level
        let riskColor, description, riskClass;
        
        if (riskLevel.toLowerCase() === 'low') {
            riskColor = 'var(--success-color)';
            description = 'Employee is likely to stay with the company (low risk of attrition). Continue with current engagement strategies.';
            riskClass = 'low-risk';
        } else if (riskLevel.toLowerCase() === 'medium') {
            riskColor = 'var(--warning-color)';
            description = 'Employee has a moderate chance of leaving. Consider additional engagement strategies to keep them satisfied.';
            riskClass = 'medium-risk';
        } else {
            riskColor = 'var(--danger-color)';
            description = 'Employee is likely to leave the company (high risk of attrition). Immediate action is recommended to improve retention.';
            riskClass = 'high-risk';
        }
        
        // Update UI
        riskLevelElement.textContent = `${riskLevel} Risk`;
        riskLevelElement.style.color = riskColor;
        riskScoreElement.style.color = riskColor;
        riskDescriptionElement.textContent = description;
        riskDescriptionElement.className = 'risk-description ' + riskClass;
        
        // Update risk factors and retention suggestions
        riskDriversElement.innerHTML = '';

        if (factors.length > 0) {
            const driversTitle = document.createElement('p');
            // Use a neutral title for all risk levels
            driversTitle.textContent = 'Factors behind this assessment:';
            driversTitle.style.fontWeight = '600';
            driversTitle.style.marginBottom = '10px';
            driversTitle.style.color = '#00bcd4';
            riskDriversElement.appendChild(driversTitle);

            const driversList = document.createElement('ul');
            driversList.style.listStyleType = 'none';
            driversList.style.padding = '0';
            driversList.style.margin = '0';

            function normalizeText(s) {
                return String(s).replace(/\b1 years\b/g, '1 year');
            }

            factors.forEach(factor => {
                const listItem = document.createElement('li');
                listItem.style.padding = '8px 0';
                listItem.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';

                const factorText = document.createElement('span');
                factorText.textContent = normalizeText(factor);
                factorText.style.color = '#00bcd4';

                listItem.appendChild(factorText);
                driversList.appendChild(listItem);
            });

            riskDriversElement.appendChild(driversList);
        }

        // Show the previous prediction (if any) under the current factors
        if (previousPrediction) {
            const prevTitle = document.createElement('p');
            prevTitle.textContent = 'Previous prediction for comparison (previous input):';
            prevTitle.className = 'previous-prediction-title';
            riskDriversElement.appendChild(prevTitle);

            const prevSummary = document.createElement('div');
            prevSummary.className = 'previous-prediction';

            const levelText = document.createElement('div');
            levelText.textContent = `Risk level: ${previousPrediction.risk_level} (score: ${previousPrediction.risk.toFixed(3)})`;
            prevSummary.appendChild(levelText);

            if (Array.isArray(previousPrediction.factors) && previousPrediction.factors.length > 0) {
                const isPersonalFactor = (s) => /Employee is|Current monthly compensation|Overtime|Years at the company/i.test(String(s));
                const filtered = previousPrediction.factors.filter(f => !isPersonalFactor(f)).slice(0, 3);
                if (filtered.length > 0) {
                    const prevFactorsList = document.createElement('ul');
                    prevFactorsList.className = 'previous-prediction-factors';
                    filtered.forEach(f => {
                        const li = document.createElement('li');
                        li.textContent = normalizeText(f);
                        li.className = 'previous-prediction-factor-item';
                        prevFactorsList.appendChild(li);
                    });
                    prevSummary.appendChild(prevFactorsList);
                }
            }

            riskDriversElement.appendChild(prevSummary);
        }

        if (retentionSuggestions.length > 0) {
            const suggestionsTitle = document.createElement('p');
            suggestionsTitle.textContent = 'Suggested Retention Actions:';
            suggestionsTitle.className = 'retention-suggestions-title';
            riskDriversElement.appendChild(suggestionsTitle);

            const suggestionsList = document.createElement('ul');
            suggestionsList.className = 'retention-suggestions';

            retentionSuggestions.forEach(item => {
                const li = document.createElement('li');
                li.className = 'retention-suggestion-item';
                li.textContent = item;
                suggestionsList.appendChild(li);
            });

            riskDriversElement.appendChild(suggestionsList);
        }

        // After rendering current results, store them as the previous prediction for the next run
        previousPrediction = {
            risk: Number.isFinite(riskScore) ? riskScore : 0,
            risk_level: riskLevel,
            factors: Array.isArray(factors) ? [...factors] : []
        };
        
        // Add timestamp if available
        if (data.timestamp) {
            const timestamp = new Date(data.timestamp).toLocaleString();
            const timeElement = document.createElement('div');
            timeElement.style.marginTop = '15px';
            timeElement.style.fontSize = '12px';
            timeElement.style.color = 'rgba(95, 7, 7, 0.6)';
            timeElement.textContent = `Prediction made on: ${timestamp}`;
            riskDriversElement.appendChild(timeElement);
        }
        
        // Show results with fade-in effect
        resultCard.style.display = 'block';
        resultCard.style.opacity = '0';
        resultCard.style.transition = 'opacity 0.5s ease';
        
        // Trigger reflow
        void resultCard.offsetWidth;
        
        // Fade in
        resultCard.style.opacity = '1';
        
        // Smooth scroll to results
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // Helper function to animate the risk score counter
    function animateValue(element, start, end, duration) {
        const range = end - start;
        const minFrameTime = 50; // 50ms per frame
        const totalFrames = Math.round(duration / minFrameTime);
        let frame = 0;
        
        const counter = setInterval(() => {
            frame++;
            
            // Calculate current value with easing function (easeOutQuad)
            const progress = frame / totalFrames;
            const current = start + (range * (1 - Math.pow(1 - progress, 3)));
            
            // Update the element
            element.textContent = current.toFixed(3);
            
            // Stop the animation when done
            if (frame === totalFrames) {
                clearInterval(counter);
                element.textContent = end.toFixed(3);
            }
        }, minFrameTime);
    }

    async function findSimilar(formData) {
        try {
            const response = await fetch('/api/find_similar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    age: formData.age,
                    jobRole: formData.jobRole,
                    monthlyIncome: formData.monthlyIncome,
                    yearsAtCompany: formData.yearsAtCompany,
                    overTime: formData.overTime,
                    k: 5
                })
            });
            if (!response.ok) {
                let msg = 'Similar employees unavailable';
                try {
                    const errData = await response.json();
                    if (errData && errData.error) msg = errData.error;
                } catch (_) {}
                const note = document.createElement('p');
                note.textContent = `Similar employees not available: ${msg}. Upload the dataset to enable this feature.`;
                note.className = 'similar-employees-title';
                riskDriversElement.appendChild(note);
                console.error('findSimilar error:', msg);
                return;
            }
            const data = await response.json();
            if (!data || !Array.isArray(data.similar) || data.similar.length === 0) {
                const note = document.createElement('p');
                note.textContent = 'No similar employees found for the given inputs.';
                note.className = 'similar-employees-title';
                riskDriversElement.appendChild(note);
                return;
            }
            const title = document.createElement('p');
            title.textContent = 'Similar employees in the dataset:';
            title.className = 'similar-employees-title';
            riskDriversElement.appendChild(title);

            const list = document.createElement('ul');
            list.className = 'similar-employees';

            data.similar.forEach(item => {
                const li = document.createElement('li');
                li.className = 'similar-employee-item';
                const parts = [];
                if (typeof item.EmployeeNumber !== 'undefined') parts.push(`#${item.EmployeeNumber}`);
                if (typeof item.Age !== 'undefined' && item.Age !== null) parts.push(`Age ${item.Age}`);
                if (typeof item.YearsAtCompany !== 'undefined' && item.YearsAtCompany !== null) parts.push(`${item.YearsAtCompany} years`);
                if (typeof item.MonthlyIncome !== 'undefined' && item.MonthlyIncome !== null) parts.push(`Income ${item.MonthlyIncome}`);
                if (typeof item.JobRole === 'string') parts.push(String(item.JobRole));
                if (typeof item.Attrition === 'string') parts.push(`Attrition: ${item.Attrition}`);
                li.textContent = parts.join(' • ');
                list.appendChild(li);
            });

            riskDriversElement.appendChild(list);
        } catch (err) {
            return;
        }
    }
});
