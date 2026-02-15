let sec = 0;
let timer = null;

function startTimer() {
    timer = setInterval(() => {
        sec++;
        let m = Math.floor(sec / 60);
        let s = sec % 60;
        document.getElementById('displayTime').innerText =
            `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;

        if (sec >= 480 && sec < 600) {
            document.getElementById('timerContainer').classList.add('timer-warning');
            document.getElementById('timerIcon').classList.replace('fa-hourglass-start', 'fa-hourglass-half');
        } else if (sec >= 600) {
            clearInterval(timer);
            document.getElementById('timerContainer').classList.replace('timer-warning', 'timer-stop');
            document.getElementById('displayTime').innerText = "10:00";
            document.getElementById('timerIcon').classList.replace('fa-hourglass-half', 'fa-hourglass-end');
        }
    }, 1000);
}

function openModal(modalId) {
    document.getElementById(modalId).style.display = "block";
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = "none";
}

window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = "none";
    }
};

const improvementCtx = document.getElementById('improvementChart').getContext('2d');
new Chart(improvementCtx, {
    type: 'bar',
    data: {
        labels: ['Dataset Size', 'Feature Count', 'Significant Features', 'Accuracy (%)'],
        datasets: [{
            label: 'Phase 1',
            data: [28, 8, 1, 72],
            backgroundColor: '#475569',
            borderRadius: 8
        }, {
            label: 'Phase 2',
            data: [600, 37, 22, 95],
            backgroundColor: '#3b82f6',
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#cbd5e1', font: { size: 12, weight: 'bold' } }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: '#ffffff05' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#94a3b8', font: { size: 11 } }
            }
        }
    }
});

const lenis = new Lenis({
    duration: 1.15,
    smoothWheel: true,
    smoothTouch: false,
    wheelMultiplier: 0.9
});

function raf(time) {
    lenis.raf(time);
    requestAnimationFrame(raf);
}

requestAnimationFrame(raf);
