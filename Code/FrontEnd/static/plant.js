// script.js

function submitForm() {
    const input = document.getElementById('image');
    const file = input.files[0];

    if (file) {
        const fileType = file.type;
        if (fileType === 'image/jpeg' || fileType === 'image/png' || fileType === 'image/jpg') {
            const formData = new FormData();
            formData.append('image', file);

            fetch('/api/predict', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);

                // Update Identification Result
                const identificationResult = document.getElementById('identificationResultContent');
                const identificationImage = document.getElementById('identificationImage');
                identificationResult.innerHTML = '<p>Plant Name: ' + data.prediction_result.plant_name + '</p><p>Disease Name: ' + data.prediction_result.disease_name + '</p>';
                identificationImage.src = 'data:image/jpeg;base64,' + data.prediction_result.image;

                // Update Detection Result
                const detectionResult = document.getElementById('detectionResult');
                const detectionImage = document.getElementById('detectionImage');

                // Clear existing content
                detectionResult.innerHTML = '';

                // Add new content
                const detectionResultContent = document.createElement('div');
                detectionResultContent.innerHTML = ' <p>Detection Result:</p><img src="' + data.detection_result_path + '" alt="Detection Image">';

                // Append elements to the container
                detectionResult.appendChild(detectionResultContent);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        } else {
            alert('File type not supported. Please upload a PNG, JPEG, or JPG file.');
        }
    } else {
        alert('Please select an image before submitting.');
    }
}


const languages={
    'en':{
        'Upload Plant Image':'Upload Plant Image',
        'Identify and Detect':'Identify and Detect',
        'Identification Result:':'Identification Result:',
        'Detection Result:':'Detection Result:'
    },
    'mr':{
        'Upload Plant Image':'वनस्पती प्रतिमा अपलोड करा',
        'Identify and Detect':'ओळखा आणि शोधा',
        'Identification Result:':'ओळख परिणाम:',
        'Detection Result:':'शोध परिणाम:'

    },
    'hi':{
        'Upload Plant Image':'पौधे की छवि अपलोड करें',
        'Identify and Detect':'पहचानें और पता लगाएं',
        'Identification Result:':'पहचान परिणाम:',
        'Detection Result:':'पता लगाने का परिणाम:'
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
    }