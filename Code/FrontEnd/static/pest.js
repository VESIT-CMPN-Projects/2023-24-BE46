// script.js

function submitForm() {
    const input = document.getElementById('image');
    const file = input.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/api/predict1', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);

            // Update Identification Result
            const identificationResult = document.getElementById('identificationResultContent');
            const identificationImage = document.getElementById('identificationImage');

            if (data && data.prediction_result && data.prediction_result.pest_name) {
                identificationResult.innerHTML = '<p>Plant Name: ' + data.prediction_result.plant_name + '</p><p>Pest Name: ' + data.prediction_result.pest_name + '</p>';
                identificationImage.src = 'data:image/jpeg;base64,' + data.prediction_result.image;
            } else {
                identificationResult.innerHTML = '<p>Unable to retrieve identification result.</p>';
                identificationImage.src = '';  // Clear image source
            }

            // Update Detection Result
            const detectionResult = document.getElementById('detectionResult');

            if (data && data.detection_result_path) {
                // Clear existing content
                detectionResult.innerHTML = '';

                // Add new content
                const detectionResultContent = document.createElement('div');
                detectionResultContent.innerHTML = ' <p>Detection Result:</p><img src="' + data.detection_result_path + '" alt="Detection Image">';

                // Append elements to the container
                detectionResult.appendChild(detectionResultContent);
            } else {
                detectionResult.innerHTML = '<p>Unable to retrieve detection result.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Please select an image before submitting.');
    }
}

// ... (existing code)

function showPesticides() {
    const identificationResult = document.getElementById('identificationResultContent');
    const identifiedPest = identificationResult.querySelector('p:nth-child(2)').textContent.split('Pest Name: ')[1].trim();

    fetch('/api/pesticides/' + identifiedPest)
        .then(response => response.json())
        .then(data => {
            const pesticidesResult = document.getElementById('pesticidesResult');

            if (data && data.pesticides) {
                // Clear existing content
                pesticidesResult.innerHTML = '';

                // Add new content
                const pesticidesResultContent = document.createElement('div');
                pesticidesResultContent.innerHTML = `<p>Recommended Pesticides for ${identifiedPest}:</p>`;

                // Create a container for pesticides and images with horizontal flex display
                const pesticidesContainer = document.createElement('div');
                pesticidesContainer.style.display = 'flex';

                // Iterate over pesticides and images
                data.pesticides.forEach(pesticide => {
                    // Create a container for each pesticide with text and image
                    const pesticideContainer = document.createElement('div');
                    pesticideContainer.style.marginRight = '20px'; // Adjust spacing as needed

                    // Display pesticide name
                    const nameElement = document.createElement('p');
                    nameElement.textContent = `${pesticide.name}`;
                    pesticideContainer.appendChild(nameElement);

                    // Display pesticide image with link
                    const img = document.createElement('img');
                    img.src = pesticide.image;
                    img.alt = `${pesticide.name} Image`;

                    const link = document.createElement('a');
                    link.href = pesticide.url;
                    link.appendChild(img);

                    pesticideContainer.appendChild(link);

                    // Append the pesticide container to the main container
                    pesticidesContainer.appendChild(pesticideContainer);
                });

                // Append the main container to the result content
                pesticidesResultContent.appendChild(pesticidesContainer);
                pesticidesResult.appendChild(pesticidesResultContent);
            } else {
                pesticidesResult.innerHTML = '<p>No pesticides information available.</p>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
const languages={
    'en':{
        'Upload Plant Image':'Upload Plant Image',
        'Identify and Detect':'Identify and Detect',
        'Identification Result:':'Identification Result:',
        'Detection Result:':'Detection Result:',
        'Recommend Pesticides':'Recommend Pesticides'
    },
    'mr':{
        'Upload Plant Image':'वनस्पती प्रतिमा अपलोड करा',
        'Identify and Detect':'ओळखा आणि शोधा',
        'Identification Result:':'ओळख परिणाम:',
        'Detection Result:':'शोध परिणाम:',
        'Recommend Pesticides':'कीटकनाशकांची शिफारस करा'

    },
    'hi':{
        'Upload Plant Image':'पौधे की छवि अपलोड करें',
        'Identify and Detect':'पहचानें और पता लगाएं',
        'Identification Result:':'पहचान परिणाम:',
        'Detection Result:':'पता लगाने का परिणाम:',
        'Recommend Pesticides':'कीटनाशकों की अनुशंसा करें'
    }
}
let currentLanguage = 'en';

    // Function to change language
    function changeLanguage(lang) {
        currentLanguage = lang;
        identify.textContent = languages[lang]['Identify and Detect'];
        upload.textContent = languages[lang]['Upload Plant Image'];
        identificationresult.textContent = languages[lang]['Identification Result:'];
        detectionresult.textContent = languages[lang]['Detection Result:'];
        pesticides.textContent = languages[lang]['Recommend Pesticides'];
    }