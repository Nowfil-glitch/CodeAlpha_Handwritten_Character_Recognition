const canvas = document.getElementById('paint-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Set up drawing canvas
ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 22;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

canvas.addEventListener('mousedown', (e) => { isDrawing = true; draw(e); });
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseleave', () => { isDrawing = false; ctx.beginPath(); });

// Touch support
canvas.addEventListener('touchstart', (e) => { isDrawing = true; draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchmove', (e) => { draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchend', () => { isDrawing = false; ctx.beginPath(); });

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('pred-digit').innerText = '?';
    document.getElementById('confidence-tag').innerText = 'Draw & click Recognize';
    renderProbabilities({});
});

function get8x8Grid() {
    // Extract pixel data from canvas and convert to 8x8 matrix (scaled 0-16)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 8;
    tempCanvas.height = 8;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(canvas, 0, 0, 8, 8);
    const imgData = tempCtx.getImageData(0, 0, 8, 8).data;
    
    const pixels = [];
    for (let i = 0; i < imgData.length; i += 4) {
        // Grayscale conversion: R/G/B average
        const avg = (imgData[i] + imgData[i+1] + imgData[i+2]) / 3.0;
        // Scale 0..255 -> 0..16
        const scaledVal = (avg / 255.0) * 16.0;
        pixels.push(scaledVal);
    }
    return pixels;
}

document.getElementById('predict-btn').addEventListener('click', async () => {
    const pixels = get8x8Grid();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('pred-digit').innerText = data.prediction;
            document.getElementById('confidence-tag').innerText = `Confidence: ${data.confidence}%`;
            renderProbabilities(data.probabilities, data.prediction);
        }
    } catch (err) {
        console.error("Error predicting character:", err);
    }
});

function renderProbabilities(probs, predDigit) {
    const container = document.getElementById('prob-grid');
    container.innerHTML = '';
    
    for (let d = 0; d <= 9; d++) {
        const val = probs[d] || 0;
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-digit" style="${d === predDigit ? 'color: #34d399;' : ''}">${d}</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width: ${val}%; ${d === predDigit ? 'background: #34d399;' : 'background: #64748b;'}"></div>
            </div>
            <span class="prob-val">${val}%</span>
        `;
        container.appendChild(row);
    }
}

// Initial empty prob grid
renderProbabilities({});
