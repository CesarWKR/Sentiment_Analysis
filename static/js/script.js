const form = document.getElementById('sentiment-form');
const resultDiv = document.getElementById('result');

const emojis = {
  Positive: '😊',
  Negative: '😠',
  Neutral: '😐'
};

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = document.getElementById('text').value;

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
      <strong><br>Probabilities:</strong>  
      <ul>  
        ${Object.entries(probs).map(([label, prob]) =>
          `<li>${label}: ${(prob * 100).toFixed(2)}%</li>`   // Format probabilities as percentages with 2 decimal places
        ).join('')}
      </ul>
    `;

    // Reset classes and widths for the progress bars
    const bars = {
      negative: document.getElementById('bar-negative'),
      neutral: document.getElementById('bar-neutral'),
      positive: document.getElementById('bar-positive')
    };

    /// Reset all bars to their initial state
    Object.entries(bars).forEach(([key, bar]) => {
      bar.classList.remove('active');
      bar.style.transition = 'none'; // Disable transition for immediate reset
      bar.style.width = '0%';
    });

    // Convert raw probabilities to lowercase keys
    const normalizedProbs = Object.fromEntries(
      Object.entries(probs).map(([key, value]) => [key.toLowerCase(), value])
    );

    // Force a reflow to reset the transition
    void bars.negative.offsetWidth;  // Trigger a reflow to reset the transition

    // Set the widths of the bars based on the probabilities
    setTimeout(() => {
      bars.negative.style.width = `${(normalizedProbs.negative * 100).toFixed(2)}%`;
      bars.neutral.style.width = `${(normalizedProbs.neutral * 100).toFixed(2)}%`;
      bars.positive.style.width = `${(normalizedProbs.positive * 100).toFixed(2)}%`;

      console.log(normalizedProbs);
      console.log(data.prediction);


      const activeBar = bars[label.toLowerCase()];
      if (activeBar) {
        activeBar.classList.add('active');
      }
    }, 100);
  } else {
    resultDiv.innerHTML = `<strong>Error:</strong> ${data.detail}`;
  }
});

