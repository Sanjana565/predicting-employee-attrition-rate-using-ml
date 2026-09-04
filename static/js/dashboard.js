document.addEventListener('DOMContentLoaded', function () {
    // Check if the canvas elements exist on this page
    const pieCanvas = document.getElementById('attritionPieChart');
    const barCanvas = document.getElementById('attritionBarChart');
    const lineCanvas = document.getElementById('attritionLineChart');

    if (!pieCanvas && !barCanvas && !lineCanvas) return; // Exit if not on dashboard page

    // Fetch analytical datasets from Flask API
    fetch('/api/analytics/charts')
        .then(response => {
            if (!response.ok) throw new Error("Failed to load analytics endpoints");
            return response.json();
        })
        .then(data => {
            renderCharts(data);
        })
        .catch(err => {
            console.error("Dashboard Analytics Load Error: ", err);
        });

    function renderCharts(analyticsData) {
        // Shared font styles
        const globalFontFamily = "'Plus Jakarta Sans', sans-serif";
        const textSecondaryColor = '#94a3b8';
        const gridBorderColor = 'rgba(255, 255, 255, 0.05)';

        // 1. Doughnut Attrition Distribution Chart
        if (pieCanvas) {
            const pieCtx = pieCanvas.getContext('2d');
            const distribution = analyticsData.attrition_distribution;
            
            new Chart(pieCtx, {
                type: 'doughnut',
                data: {
                    labels: distribution.labels,
                    datasets: [{
                        data: distribution.data,
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.85)', // Stay: Green
                            'rgba(239, 68, 68, 0.85)'   // Leave: Red
                        ],
                        borderColor: '#0f1524',
                        borderWidth: 2,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily, weight: '500' },
                                padding: 15
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(8, 12, 20, 0.95)',
                            titleFont: { family: globalFontFamily, weight: '700' },
                            bodyFont: { family: globalFontFamily },
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1
                        }
                    }
                }
            });
        }

        // 2. Bar Chart: Attrition counts by Department
        if (barCanvas) {
            const barCtx = barCanvas.getContext('2d');
            const department = analyticsData.department_attrition;

            // Generate premium purple gradient for bars
            const purpleGradient = barCtx.createLinearGradient(0, 0, 0, 250);
            purpleGradient.addColorStop(0, '#a855f7'); // Violet
            purpleGradient.addColorStop(1, 'rgba(124, 58, 237, 0.1)'); // Translucent

            new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: department.labels,
                    datasets: [{
                        label: 'Attrition Count (Likely to Leave)',
                        data: department.data,
                        backgroundColor: purpleGradient,
                        borderColor: '#8b5cf6',
                        borderWidth: 1.5,
                        borderRadius: 6,
                        barThickness: 28
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(8, 12, 20, 0.95)',
                            titleFont: { family: globalFontFamily, weight: '700' },
                            bodyFont: { family: globalFontFamily },
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily, weight: '500' }
                            }
                        },
                        y: {
                            grid: { color: gridBorderColor },
                            ticks: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily },
                                stepSize: 1,
                                beginAtZero: true
                            }
                        }
                    }
                }
            });
        }

        // 3. Line Chart: Monthly Attrition Trend
        if (lineCanvas) {
            const lineCtx = lineCanvas.getContext('2d');
            const trend = analyticsData.monthly_trend;

            // Create neon blue/purple gradients for line chart fills
            const blueGradient = lineCtx.createLinearGradient(0, 0, 0, 250);
            blueGradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
            blueGradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

            new Chart(lineCtx, {
                type: 'line',
                data: {
                    labels: trend.labels,
                    datasets: [
                        {
                            label: 'Total Predictions Conducted',
                            data: trend.predictions,
                            borderColor: '#3b82f6',
                            backgroundColor: blueGradient,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointBackgroundColor: '#3b82f6',
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Attrition Predicted',
                            data: trend.attrition,
                            borderColor: '#ef4444',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.4,
                            pointBackgroundColor: '#ef4444',
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily, weight: '500' }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(8, 12, 20, 0.95)',
                            titleFont: { family: globalFontFamily, weight: '700' },
                            bodyFont: { family: globalFontFamily },
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: gridBorderColor },
                            ticks: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily, weight: '500' }
                            }
                        },
                        y: {
                            grid: { color: gridBorderColor },
                            ticks: {
                                color: textSecondaryColor,
                                font: { family: globalFontFamily },
                                beginAtZero: true
                            }
                        }
                    }
                }
            });
        }
    }
});
