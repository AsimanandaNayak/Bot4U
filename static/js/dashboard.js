// Dashboard JavaScript

// DOM Elements
const refreshDataBtn = document.getElementById('refresh-data');
const sidebarLinks = document.querySelectorAll('.sidebar-menu-link');
const tabContents = document.querySelectorAll('.tab-content');
const bookingStatusTabs = document.querySelectorAll('[data-booking-status]');
const recentBookingsBody = document.getElementById('recent-bookings-body');
const bookingsTableBody = document.getElementById('bookings-table-body');
const guestsTableBody = document.getElementById('guests-table-body');
const gstReportsContainer = document.getElementById('gst-reports-container');
const roomAvailabilityContainer = document.getElementById('room-availability');
const viewAllBookingsBtn = document.getElementById('view-all-bookings');
const downloadModal = document.getElementById('download-modal');
const downloadGuestName = document.getElementById('download-guest-name');
const downloadBookingDetails = document.getElementById('download-booking-details');
const cancelDownloadBtn = document.getElementById('cancel-download');
const confirmDownloadBtn = document.getElementById('confirm-download');

// Charts
let revenueChart;
let roomDistributionChart;

// Dashboard data
let dashboardData = null;
let guestsData = null;
let gstReportsData = null;

// Current email for download
let currentDownloadEmail = '';
let currentGuestName = '';
let activeBookingTab = 'all';

// Format currency
const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(amount);
};

// Format date
const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
};

// Format time
const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
};

// Create or update revenue chart
const createRevenueChart = (data) => {
    const ctx = document.getElementById('revenue-chart').getContext('2d');
    
    // Destroy existing chart if exists
    if (revenueChart) {
        revenueChart.destroy();
    }
    
    // Get data for last 7 days
    const days = Object.keys(data.revenue_by_day);
    const revenues = Object.values(data.revenue_by_day);
    
    // Create chart
    revenueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days.map(day => formatDate(day)),
            datasets: [{
                label: 'Revenue',
                data: revenues,
                borderColor: '#C0A080',
                backgroundColor: 'rgba(192, 160, 128, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return formatCurrency(context.raw);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            }
        }
    });
};

// Create or update room distribution chart
const createRoomDistributionChart = (data) => {
    const ctx = document.getElementById('room-distribution-chart').getContext('2d');
    
    // Destroy existing chart if exists
    if (roomDistributionChart) {
        roomDistributionChart.destroy();
    }
    
    // Get room type data
    const roomTypes = Object.keys(data.room_types);
    const roomCounts = Object.values(data.room_types);
    
    // Define colors for each room type
    const colors = {
        'Basic': '#4CAF50',
        'Comfort': '#2196F3',
        'Luxury': '#C0A080'
    };
    
    // Create chart
    roomDistributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: roomTypes,
            datasets: [{
                data: roomCounts,
                backgroundColor: roomTypes.map(type => colors[type] || '#757575'),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
};

// Render room availability cards
const renderRoomAvailability = (data) => {
    roomAvailabilityContainer.innerHTML = '';
    
    const roomTypes = Object.keys(data.room_availability_by_type);
    
    // Define icons for each room type
    const icons = {
        'Basic': 'fa-bed',
        'Comfort': 'fa-couch',
        'Luxury': 'fa-gem'
    };
    
    roomTypes.forEach(type => {
        const availabilityData = data.room_availability_by_type[type];
        const percentage = availabilityData.percentage;
        
        const roomCard = document.createElement('div');
        roomCard.className = 'room-type-card';
        roomCard.innerHTML = `
            <div class="room-type-icon">
                <i class="fas ${icons[type] || 'fa-home'}"></i>
            </div>
            <div class="room-type-name">${type}</div>
            <div class="room-type-availability">
                ${availabilityData.available} of ${availabilityData.total} Available
            </div>
            <div class="progress-bar">
                <div class="progress" style="width: ${percentage}%"></div>
            </div>
            <div class="progress-text">${percentage}% Available</div>
        `;
        
        roomAvailabilityContainer.appendChild(roomCard);
    });
};

// Render recent bookings table
const renderRecentBookings = (bookings) => {
    recentBookingsBody.innerHTML = '';
    
    if (!bookings || bookings.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="6" class="no-data">
                <div><i class="fas fa-calendar-times"></i></div>
                <div class="no-data-text">No bookings found</div>
                <div class="no-data-subtext">New bookings will appear here</div>
            </td>
        `;
        recentBookingsBody.appendChild(emptyRow);
        return;
    }
    
    bookings.forEach(booking => {
        const row = document.createElement('tr');
        
        const statusClass = booking.status === 'Active' ? 'status-active' : 'status-checked-out';
        
        row.innerHTML = `
            <td>${booking.booking_id}</td>
            <td>${booking.name}</td>
            <td>${booking.room_type} - ${booking.room_number}</td>
            <td>${formatDateTime(booking.booking_date)}</td>
            <td><span class="table-status ${statusClass}">${booking.status}</span></td>
            <td>${booking.total_amount ? formatCurrency(booking.total_amount) : 'Pending'}</td>
        `;
        
        recentBookingsBody.appendChild(row);
    });
};

// Render bookings table
const renderBookingsTable = (bookings, status = 'all') => {
    bookingsTableBody.innerHTML = '';
    
    // Filter bookings by status if needed
    let filteredBookings = bookings;
    if (status !== 'all') {
        filteredBookings = bookings.filter(booking => booking.status.toLowerCase() === status.toLowerCase());
    }
    
    if (!filteredBookings || filteredBookings.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="10" class="no-data">
                <div><i class="fas fa-calendar-times"></i></div>
                <div class="no-data-text">No bookings found</div>
                <div class="no-data-subtext">No ${status === 'all' ? '' : status} bookings available</div>
            </td>
        `;
        bookingsTableBody.appendChild(emptyRow);
        return;
    }
    
    filteredBookings.forEach(booking => {
        const row = document.createElement('tr');
        
        const statusClass = booking.status === 'Active' ? 'status-active' : 'status-checked-out';
        
        row.innerHTML = `
            <td>${booking.booking_id}</td>
            <td>${booking.name}</td>
            <td>
                <div>${booking.email}</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">${booking.phone}</div>
            </td>
            <td>${booking.room_type}</td>
            <td>${booking.room_number}</td>
            <td>${formatDateTime(booking.booking_date)}</td>
            <td>${booking.checkout_time ? formatDateTime(booking.checkout_time) : 'N/A'}</td>
            <td><span class="table-status ${statusClass}">${booking.status}</span></td>
            <td>${booking.total_amount ? formatCurrency(booking.total_amount) : 'Pending'}</td>
            <td>
                <button class="chart-btn" onclick="viewBookingDetails('${booking.booking_id}')">
                    <i class="fas fa-eye"></i>
                </button>
                ${booking.status === 'Checked-out' ? 
                    `<button class="chart-btn" onclick="downloadGSTReport('${booking.email}', '${booking.name}')">
                        <i class="fas fa-file-download"></i>
                    </button>` : ''}
            </td>
        `;
        
        bookingsTableBody.appendChild(row);
    });
};

// Render guests table
const renderGuestsTable = (guests) => {
    guestsTableBody.innerHTML = '';
    
    if (!guests || guests.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="7" class="no-data">
                <div><i class="fas fa-users"></i></div>
                <div class="no-data-text">No guests found</div>
                <div class="no-data-subtext">New guests will appear here</div>
            </td>
        `;
        guestsTableBody.appendChild(emptyRow);
        return;
    }
    
    // Group guests by email
    const groupedGuests = {};
    
    guests.forEach(guest => {
        const email = guest.email;
        
        if (!groupedGuests[email]) {
            groupedGuests[email] = {
                name: guest.name,
                email: guest.email,
                phone: guest.phone,
                bookings: [],
                totalSpent: 0
            };
        }
        
        groupedGuests[email].bookings.push({
            bookingDate: guest.booking_date,
            checkoutTime: guest.checkout_time,
            roomType: guest.room_type,
            status: guest.status,
            totalAmount: guest.total_amount || 0
        });
        
        if (guest.total_amount) {
            groupedGuests[email].totalSpent += guest.total_amount;
        }
    });
    
    // Convert to array and sort by total spent
    const guestsList = Object.values(groupedGuests).sort((a, b) => b.totalSpent - a.totalSpent);
    
    guestsList.forEach(guest => {
        const row = document.createElement('tr');
        
        // Get the most recent stay
        const lastStay = guest.bookings.sort((a, b) => {
            return new Date(b.bookingDate) - new Date(a.bookingDate);
        })[0];
        
        row.innerHTML = `
            <td>${guest.name}</td>
            <td>${guest.email}</td>
            <td>${guest.phone}</td>
            <td>${formatDateTime(lastStay.bookingDate)}</td>
            <td>${guest.bookings.length}</td>
            <td>${formatCurrency(guest.totalSpent)}</td>
            <td>
                <button class="chart-btn" onclick="viewGuestDetails('${guest.email}')">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="chart-btn" onclick="downloadGuestGSTReports('${guest.email}', '${guest.name}')">
                    <i class="fas fa-file-download"></i>
                </button>
            </td>
        `;
        
        guestsTableBody.appendChild(row);
    });
};

// Render GST reports
const renderGSTReports = (reports) => {
    gstReportsContainer.innerHTML = '';
    
    if (!reports || reports.length === 0) {
        gstReportsContainer.innerHTML = `
            <div class="no-data">
                <div><i class="fas fa-file-invoice"></i></div>
                <div class="no-data-text">No GST reports found</div>
                <div class="no-data-subtext">GST reports for checked-out guests will appear here</div>
            </div>
        `;
        return;
    }
    
    reports.forEach(report => {
        const reportCard = document.createElement('div');
        reportCard.className = 'gst-report-card';
        
        reportCard.innerHTML = `
            <div class="gst-report-header">
                <div class="gst-report-title">GST Invoice: ${report.booking_id}</div>
                <div class="gst-report-date">Checkout: ${formatDateTime(report.checkout_time)}</div>
            </div>
            <div class="gst-report-details">
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">Guest</div>
                    <div class="gst-report-detail-value">${report.name}</div>
                </div>
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">Stay Duration</div>
                    <div class="gst-report-detail-value">${report.days} Days</div>
                </div>
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">Room</div>
                    <div class="gst-report-detail-value">${report.room_type} - ${report.room_number}</div>
                </div>
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">Base Amount</div>
                    <div class="gst-report-detail-value">${formatCurrency(report.base_amount)}</div>
                </div>
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">GST (18%)</div>
                    <div class="gst-report-detail-value">${formatCurrency(report.gst_amount)}</div>
                </div>
                <div class="gst-report-detail">
                    <div class="gst-report-detail-label">Total Amount</div>
                    <div class="gst-report-detail-value">${formatCurrency(report.total_amount)}</div>
                </div>
            </div>
            <div class="gst-report-actions">
                <button class="gst-report-btn" onclick="downloadGSTReport('${report.email}', '${report.name}', '${report.booking_id}')">
                    <i class="fas fa-download"></i> Download Report
                </button>
            </div>
        `;
        
        gstReportsContainer.appendChild(reportCard);
    });
};

// Function to fetch dashboard data
const fetchDashboardData = async () => {
    try {
        const response = await fetch('/dashboard/data');
        
        if (!response.ok) {
            throw new Error('Failed to fetch dashboard data');
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
        alert('Failed to load dashboard data. Please try again.');
        return null;
    }
};

// Function to fetch guests data
const fetchGuestsData = async () => {
    try {
        const response = await fetch('/dashboard/guests');
        
        if (!response.ok) {
            throw new Error('Failed to fetch guests data');
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching guests data:', error);
        alert('Failed to load guests data. Please try again.');
        return null;
    }
};

// Function to fetch GST reports data
const fetchGSTReportsData = async () => {
    try {
        const response = await fetch('/dashboard/gst-reports');
        
        if (!response.ok) {
            throw new Error('Failed to fetch GST reports');
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching GST reports:', error);
        alert('Failed to load GST reports. Please try again.');
        return null;
    }
};

// Function to update dashboard stats
const updateDashboardStats = (data) => {
    document.getElementById('total-bookings').textContent = data.stats.total_bookings;
    document.getElementById('active-bookings').textContent = data.stats.active_bookings;
    document.getElementById('total-revenue').textContent = formatCurrency(data.stats.total_revenue);
    document.getElementById('total-gst').textContent = formatCurrency(data.stats.total_gst);
    
    // Placeholder trends (could be calculated from actual data in a real implementation)
    document.getElementById('bookings-trend').textContent = '5%';
    document.getElementById('revenue-trend').textContent = '8%';
};

// Function to refresh all dashboard data
const refreshDashboard = async () => {
    // Show loading indicator
    const loader = document.createElement('div');
    loader.className = 'loader-container';
    loader.innerHTML = '<div class="loader"></div>';
    document.body.appendChild(loader);
    
    try {
        // Fetch all data in parallel
        const [dashboardDataResult, guestsDataResult, gstReportsDataResult] = await Promise.all([
            fetchDashboardData(),
            fetchGuestsData(),
            fetchGSTReportsData()
        ]);
        
        // Update global variables
        dashboardData = dashboardDataResult;
        guestsData = guestsDataResult;
        gstReportsData = gstReportsDataResult;
        
        if (dashboardData) {
            // Update dashboard stats
            updateDashboardStats(dashboardData);
            
            // Create charts
            createRevenueChart(dashboardData);
            createRoomDistributionChart(dashboardData);
            
            // Render room availability
            renderRoomAvailability(dashboardData);
            
            // Render recent bookings
            renderRecentBookings(dashboardData.recent_bookings);
        }
        
        // Render other data if current tab requires it
        const activeTab = document.querySelector('.tab-content.active').id;
        
        if (activeTab === 'bookings-tab' && guestsData) {
            renderBookingsTable(guestsData, activeBookingTab);
        }
        
        if (activeTab === 'guests-tab' && guestsData) {
            renderGuestsTable(guestsData);
        }
        
        if (activeTab === 'gst-reports-tab' && gstReportsData) {
            renderGSTReports(gstReportsData);
        }
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
        alert('Failed to refresh dashboard data. Please try again.');
    } finally {
        // Remove loading indicator
        document.body.removeChild(loader);
    }
};

// Function to handle tab switching
const switchTab = (tabId) => {
    // Hide all tabs
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(`${tabId}-tab`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Update sidebar links
    sidebarLinks.forEach(link => {
        link.classList.remove('active');
        
        if (link.getAttribute('data-tab') === tabId) {
            link.classList.add('active');
        }
    });
    
    // Load data for the tab if needed
    if (tabId === 'bookings' && guestsData) {
        renderBookingsTable(guestsData, activeBookingTab);
    } else if (tabId === 'guests' && guestsData) {
        renderGuestsTable(guestsData);
    } else if (tabId === 'gst-reports' && gstReportsData) {
        renderGSTReports(gstReportsData);
    }
};

// Function to download GST report
const downloadGSTReport = (email, name, bookingId = '') => {
    currentDownloadEmail = email;
    currentGuestName = name;
    
    // Update modal content
    downloadGuestName.textContent = name;
    downloadBookingDetails.textContent = bookingId ? `Booking ID: ${bookingId}` : 'All GST reports';
    
    // Show modal
    downloadModal.classList.add('show');
};

// Function to download all GST reports for a guest
const downloadGuestGSTReports = (email, name) => {
    downloadGSTReport(email, name);
};

// Function to view booking details (placeholder)
const viewBookingDetails = (bookingId) => {
    alert(`View details for booking ${bookingId}`);
    // In a real implementation, this would open a modal with booking details
};

// Function to view guest details (placeholder)
const viewGuestDetails = (email) => {
    alert(`View details for guest with email ${email}`);
    // In a real implementation, this would open a modal with guest details
};

// Function to handle booking status tab selection
const selectBookingStatusTab = (status) => {
    // Update active status
    activeBookingTab = status;
    
    // Update tab UI
    bookingStatusTabs.forEach(tab => {
        tab.classList.remove('active');
        
        if (tab.getAttribute('data-booking-status') === status) {
            tab.classList.add('active');
        }
    });
    
    // Render bookings with selected status
    if (guestsData) {
        renderBookingsTable(guestsData, status);
    }
};

// Function to initialize the dashboard
const initDashboard = async () => {
    // Initial data load
    await refreshDashboard();
    
    // Add event listeners
    refreshDataBtn.addEventListener('click', refreshDashboard);
    
    // Tab navigation
    sidebarLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = link.getAttribute('data-tab');
            if (tabId) {
                switchTab(tabId);
            }
        });
    });
    
    // Booking status tabs
    bookingStatusTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const status = tab.getAttribute('data-booking-status');
            selectBookingStatusTab(status);
        });
    });
    
    // View all bookings button
    viewAllBookingsBtn.addEventListener('click', () => {
        switchTab('bookings');
    });
    
    // Download modal events
    cancelDownloadBtn.addEventListener('click', () => {
        downloadModal.classList.remove('show');
    });
    
    confirmDownloadBtn.addEventListener('click', () => {
        // Redirect to download endpoint
        window.location.href = `/download_gst/${currentDownloadEmail.replace('@', '_at_')}`;
        downloadModal.classList.remove('show');
    });
    
    // Close modal when clicking outside
    downloadModal.addEventListener('click', (e) => {
        if (e.target === downloadModal) {
            downloadModal.classList.remove('show');
        }
    });
    
    // Chart period buttons
    document.querySelectorAll('.chart-btn[data-period]').forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all period buttons
            document.querySelectorAll('.chart-btn[data-period]').forEach(b => {
                b.classList.remove('active');
            });
            
            // Add active class to clicked button
            btn.classList.add('active');
            
            // In a real implementation, this would update the chart data for the selected period
            // For this demo, we'll just show a message
            alert(`Selected period: ${btn.getAttribute('data-period')}`);
        });
    });
};

// Call init function when DOM is loaded
document.addEventListener('DOMContentLoaded', initDashboard);
