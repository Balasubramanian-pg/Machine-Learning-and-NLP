document.addEventListener('DOMContentLoaded', function() {
    // Sales by Category Pie Chart
    const salesByCategoryCtx = document.getElementById('sales-by-category-chart').getContext('2d');
    const salesByCategoryChart = new Chart(salesByCategoryCtx, {
        type: 'pie',
        data: {
            labels: ['Electronics', 'Clothing', 'Home Goods'],
            datasets: [{
                label: 'Sales by Category',
                data: [300000, 150000, 50000],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
        }
    });

    // Sales by Region Bar Chart
    const salesByRegionCtx = document.getElementById('sales-by-region-chart').getContext('2d');
    const salesByRegionChart = new Chart(salesByRegionCtx, {
        type: 'bar',
        data: {
            labels: ['North', 'South', 'East', 'West'],
            datasets: [{
                label: 'Sales by Region',
                data: [120000, 100000, 150000, 130000],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(200, 99, 132, 0.8)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)',
                     'rgba(200, 99, 132, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});
