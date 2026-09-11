document.addEventListener('DOMContentLoaded', () => {

    // ── DDC Class metadata — colors and short names ─────────────────
    const DDC_META = {
        '000': { color: '#4A90D9', name: 'Computing & Information' },
        '100': { color: '#9B59B6', name: 'Philosophy & Psychology' },
        '200': { color: '#D4A017', name: 'Religion' },
        '300': { color: '#E74C3C', name: 'Social Sciences' },
        '400': { color: '#1ABC9C', name: 'Language' },
        '500': { color: '#27AE60', name: 'Science' },
        '600': { color: '#E67E22', name: 'Technology' },
        '700': { color: '#E91E63', name: 'Arts & Recreation' },
        '800': { color: '#8E44AD', name: 'Literature' },
        '900': { color: '#34495E', name: 'History & Geography' },
    };

    // ── Tab Switching ───────────────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.closest('.page-section');
            section.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            section.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // ── File Upload (Drag & Drop) ──────────────────────────────────
    const sections = ['front', 'summary', 'index'];
    const filesData = { front: null, summary: null, index: null };

    sections.forEach(sec => {
        const dropZone = document.getElementById(`drop-${sec}`);
        const fileInput = document.getElementById(`file-${sec}`);
        const imgPreview = dropZone.querySelector('.preview-img');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(sec, e.dataTransfer.files[0], imgPreview);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(sec, e.target.files[0], imgPreview);
            }
        });
    });

    function handleFile(section, file, imgPreview) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        filesData[section] = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imgPreview.src = e.target.result;
            imgPreview.removeAttribute('hidden');
        };
        reader.readAsDataURL(file);
    }

    // ── Camera Capture ─────────────────────────────────────────────
    const streams = {};
    document.querySelectorAll('.start-camera-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const sec = btn.getAttribute('data-target');
            const video = document.getElementById(`video-${sec}`);
            const captureBtn = document.querySelector(`.capture-btn[data-target="${sec}"]`);
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                video.srcObject = stream;
                video.removeAttribute('hidden');
                btn.setAttribute('hidden', 'true');
                captureBtn.removeAttribute('hidden');
                streams[sec] = stream;
            } catch (err) {
                console.error("Camera access denied or unavailable", err);
                alert("Could not access camera. Please check permissions or use file upload.");
            }
        });
    });

    document.querySelectorAll('.capture-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sec = btn.getAttribute('data-target');
            const video = document.getElementById(`video-${sec}`);
            const canvas = document.getElementById(`canvas-${sec}`);
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            video.setAttribute('hidden', 'true');
            canvas.removeAttribute('hidden');
            btn.textContent = "Retake Photo";
            
            // Stop stream
            if (streams[sec]) {
                streams[sec].getTracks().forEach(track => track.stop());
                streams[sec] = null;
            }
            
            // Convert canvas to Blob
            canvas.toBlob(blob => {
                const file = new File([blob], `camera_${sec}.jpg`, { type: 'image/jpeg' });
                filesData[sec] = file;
            }, 'image/jpeg', 0.95);
            
            // Reset button to allow restart
            btn.onclick = () => {
                canvas.setAttribute('hidden', 'true');
                btn.setAttribute('hidden', 'true');
                btn.textContent = "Capture Photo";
                document.querySelector(`.start-camera-btn[data-target="${sec}"]`).removeAttribute('hidden');
            };
        });
    });

    // ── Form Submission ────────────────────────────────────────────
    document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!filesData.front) {
            alert("Front Page / Cover image is required.");
            return;
        }

        const formData = new FormData();
        formData.append('front_page', filesData.front);
        
        if (filesData.summary) formData.append('summary_page', filesData.summary);
        if (filesData.index) formData.append('index_page', filesData.index);

        document.getElementById('loader').removeAttribute('hidden');
        document.getElementById('results-pane').setAttribute('hidden', 'true');

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || "Failed to process images.");
            }
            
            if (data.error) {
                alert(`Pipeline Error: ${data.error}`);
            }

            displayResults(data);
        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
        } finally {
            document.getElementById('loader').setAttribute('hidden', 'true');
        }
    });

    // ── Display Results ────────────────────────────────────────────
    function displayResults(data) {
        document.getElementById('results-pane').removeAttribute('hidden');
        
        // Predicted class with color
        const classEl = document.getElementById('res-class');
        classEl.textContent = data.predicted_class || "N/A";
        if (data.predicted_class && DDC_META[data.predicted_class]) {
            const color = DDC_META[data.predicted_class].color;
            classEl.style.background = `linear-gradient(135deg, ${color}, ${adjustColor(color, 30)})`;
            classEl.style.webkitBackgroundClip = 'text';
            classEl.style.webkitTextFillColor = 'transparent';
            classEl.style.backgroundClip = 'text';
        }
        
        document.getElementById('res-class-name').textContent = data.predicted_class_name || "";
        
        // Animated confidence display
        const confEl = document.getElementById('res-conf');
        if (data.confidence) {
            animateConfidence(confEl, data.confidence);
        } else {
            confEl.textContent = "N/A";
        }
        
        // Routing badge with color
        const routingEl = document.getElementById('res-routing');
        const routing = data.routing_outcome || "N/A";
        routingEl.textContent = routing.replace(/_/g, ' ');
        routingEl.className = 'routing-badge';
        if (routing === 'auto_accept') routingEl.classList.add('badge-accept');
        else if (routing === 'needs_review') routingEl.classList.add('badge-review');
        else routingEl.classList.add('badge-reject');
        
        // OCR engine badge
        const engineEl = document.getElementById('res-engine');
        engineEl.textContent = data.ocr_engine === 'gemini_vision' ? 'Gemini Vision' : 'Tesseract';
        
        // Ensemble strategy badge
        const ensembleEl = document.getElementById('res-ensemble');
        if (data.ensemble && data.ensemble.strategy) {
            const strategy = data.ensemble.strategy;
            ensembleEl.className = 'ensemble-badge';
            if (strategy === 'agreement') {
                ensembleEl.textContent = 'Models Agree';
                ensembleEl.classList.add('ensemble-agreement');
            } else if (strategy === 'disagreement') {
                ensembleEl.textContent = 'Models Disagree';
                ensembleEl.classList.add('ensemble-disagreement');
            } else {
                ensembleEl.textContent = 'TF-IDF Only';
                ensembleEl.classList.add('ensemble-tfidf');
            }
        } else {
            ensembleEl.textContent = '';
            ensembleEl.className = '';
        }
        
        // Extracted fields
        if (data.fields) {
            document.getElementById('res-title').textContent = data.fields.title || "--";
            document.getElementById('res-subtitle').textContent = data.fields.subtitle || "--";
            document.getElementById('res-author').textContent = data.fields.author || "--";
            document.getElementById('res-description').textContent = data.fields.description || "--";
        }
        
        // Book-spine probability chart
        buildProbChart(data.all_probabilities, data.predicted_class);
        
        document.getElementById('res-raw').textContent = data.raw_text || "--";
        
        // Smooth scroll to results
        setTimeout(() => {
            document.getElementById('results-pane').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    // ── Animated Confidence Counter ────────────────────────────────
    function animateConfidence(el, targetValue) {
        const targetPct = targetValue * 100;
        const duration = 800;
        const start = performance.now();
        
        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = eased * targetPct;
            el.textContent = `${current.toFixed(1)}%`;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }

    // ── Build Probability Chart with Book Spine Colors ─────────────
    function buildProbChart(probs, predictedClass) {
        const chartContainer = document.getElementById('prob-chart');
        chartContainer.innerHTML = '';
        
        if (!probs) return;
        
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        
        sorted.forEach(([label, prob], index) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            
            // DDC label
            const labelEl = document.createElement('span');
            labelEl.className = 'prob-label';
            labelEl.textContent = label;
            
            const meta = DDC_META[label] || { color: '#888', name: label };
            labelEl.style.color = meta.color;
            
            // Short class name
            const nameEl = document.createElement('span');
            nameEl.className = 'prob-class-name';
            nameEl.textContent = meta.name;
            
            // Bar wrapper
            const barWrap = document.createElement('div');
            barWrap.className = 'prob-bar-wrap';
            
            // Bar with class-specific color
            const bar = document.createElement('div');
            bar.className = 'prob-bar';
            const pct = prob * 100;
            
            // Animate bar width
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = `${pct}%`;
            }, 100 + index * 60);
            
            // Color the bar with the DDC class color
            if (label === predictedClass) {
                bar.classList.add('prob-bar-active');
                bar.style.background = `linear-gradient(90deg, ${meta.color}, ${adjustColor(meta.color, 20)})`;
            } else {
                bar.style.background = `${meta.color}33`;  // 20% opacity
            }
            
            // Value
            const valEl = document.createElement('span');
            valEl.className = 'prob-val';
            valEl.textContent = `${pct.toFixed(1)}%`;
            
            barWrap.appendChild(bar);
            row.appendChild(labelEl);
            row.appendChild(nameEl);
            row.appendChild(barWrap);
            row.appendChild(valEl);
            chartContainer.appendChild(row);
        });
    }

    // ── Color Utility — lighten/darken a hex color ─────────────────
    function adjustColor(hex, amount) {
        hex = hex.replace('#', '');
        const r = Math.min(255, Math.max(0, parseInt(hex.substr(0, 2), 16) + amount));
        const g = Math.min(255, Math.max(0, parseInt(hex.substr(2, 2), 16) + amount));
        const b = Math.min(255, Math.max(0, parseInt(hex.substr(4, 2), 16) + amount));
        return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
    }
});
