const form = document.getElementById('sentiment-form');
const resultDiv = document.getElementById('result');
const textarea = document.getElementById('text');
const charCount = document.getElementById("char-count");
const maxLength = 256;  // Maximum length for the textarea

if (charCount) {   // Check if charCount exists before accessing it
  charCount.classList.add("warning"); // Add initial class for styling
}

// Update character count on input
textarea.addEventListener('input', () => {
  const currentLength = textarea.value.length;
  charCount.textContent = `${currentLength} / ${maxLength}`;

  const remaining = maxLength - currentLength;

  charCount.classList.remove('blink', 'shake');  // Remove any previous animation classes
  
  if (currentLength >= maxLength) {
    charCount.style.color = 'red';  // Change color to red if exceeded
    charCount.classList.add('shake');  // Add shake class for visual feedback
  } else if (remaining <= 10) {
    charCount.style.color = 'red';  // Change color to red if less than or equal to 10 characters remaining
    charCount.classList.add('blink');  // Add blink class for visual feedback
  } else if (remaining <= 50) {
    charCount.style.color = 'orange';  // Change color to orange if less than or equal to 50 characters remaining
  } else {
    charCount.style.color = 'gray';  // Change color to gray if more than 50 characters remaining
  }
});

const emojis = {
  Positive: '😊',
  Negative: '😠',
  Neutral: '😐'
};

// Handle form submission
resultDiv.style.display = 'none';  // Initially hide the result div
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  
  const text = textarea.value;

  if (text.length > maxLength) {
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `<strong>Error:</strong> Text exceeds maximum length of ${maxLength} characters.`;
    return;
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (response.ok) {
      const label = data.prediction.label;
      const emoji = emojis[label] || '';
      const probs = data.prediction.probabilities;

      resultDiv.style.display = 'block';

      // Display the result with the label, emoji, and probabilities and render them as a list
      resultDiv.innerHTML = `
        <strong>Input:</strong> ${data.input}<br>
        <strong>Prediction:</strong> ${label} <span class="emoji">${emoji}</span><br>
      `;

      // Reset classes and widths for the progress bars
      const bars = {
        negative: {
          bar_element: document.getElementById('bar-negative'),
          label: document.getElementById('label-negative')
        },
        neutral: {
          bar_element: document.getElementById('bar-neutral'),
          label: document.getElementById('label-neutral')
        },
        positive: {
          bar_element: document.getElementById('bar-positive'),
          label: document.getElementById('label-positive')
        }
      };

      /// Reset all bars and hide percentage labels
      Object.entries(bars).forEach(([key, value]) => {
        value.bar_element.classList.remove('active');
        value.bar_element.style.transition = 'none'; // Disable transition for immediate reset
        value.bar_element.style.width = '0%';
        value.label.style.display = 'none'; // Hide the label
      });

      // Convert raw probabilities to lowercase keys
      const normalizedProbs = Object.fromEntries(
        Object.entries(probs).map(([key, value]) => [key.toLowerCase(), value])
      );

      // Force a reflow to reset the transition
      void bars.negative.offsetWidth;  // Trigger a reflow to reset the transition

      
      // Set new bar widths and show labels
      setTimeout(() => {
        let maxKey = null;
        let maxProb = -1;

        for (const [key, { bar_element, label }] of Object.entries(bars)) {   // Iterate over each bar
          const prob = normalizedProbs[key];  // Get the probability for the current bar
          if (prob !== undefined && bar_element && label) {   // Check if probability exists and elements are valid
            const percent = (prob * 100).toFixed(2);   // Calculate percentage
            console.log(`${key}: ${percent}%`);
            bar_element.style.width = `${percent}%`;   // Set the width of the bar
            label.textContent = `${percent}%`;   // Set the label text
            label.style.display = 'inline';   // Show the label

            // Check for the maximum probability
            if (prob > maxProb) {
              maxProb = prob;
              maxKey = key;  // Store the key with the maximum probability
            }
          }
        }

        // Add the 'active' class to the bar with the maximum probability
        if (maxKey && bars[maxKey]) {
          bars[maxKey].bar_element.classList.add('active');
        }
      }, 100);
    } else {
      resultDiv.innerHTML = `<strong>Error:</strong> ${data.detail}`;
    }
  } catch (error) {
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
  }
});

