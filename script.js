// Tab navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all tabs and sections
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

        // Add active class to clicked tab and corresponding section
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// Toggle content buttons
document.querySelectorAll('.toggle-details-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        const target = document.getElementById(targetId);
        target.classList.toggle('show');

        // Update button icon
        const icon = btn.querySelector('i');
        if (target.classList.contains('show')) {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-up');
        } else {
            icon.classList.remove('fa-chevron-up');
            icon.classList.add('fa-chevron-down');
        }
    });
});

// Toggle all BMI charts
function toggleAllBMICharts() {
    const charts = ['bmi-charts', 'bmi-age-chart'];
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        chart.classList.toggle('show');
    });
}

// Print function
function printReport() {
    window.print();
}