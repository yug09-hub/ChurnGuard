/**
 * ChurnGuard AI - Client-Side Application
 * ========================================
 * Handles all dashboard interactivity: KPI counters, Chart.js graphs,
 * prediction form, customer explorer, and modal views.
 */

// ── Global State ─────────────────────────────────────────────
let statsData = null;
let currentPage = 1;
let searchTimeout = null;

// ── Initialize ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    startClock();
    loadStats();
    loadCustomers();
    setupNavigation();
    setupPredictionForm();
    setupCustomerSearch();
    setupModal();
});

// ── Clock ────────────────────────────────────────────────────
function startClock() {
    const el = document.getElementById("navClock");
    function tick() {
        const now = new Date();
        el.textContent = now.toLocaleTimeString("en-US", {
            hour: "2-digit", minute: "2-digit", second: "2-digit"
        });
    }
    tick();
    setInterval(tick, 1000);
}

// ── Navigation ───────────────────────────────────────────────
function setupNavigation() {
    const links = document.querySelectorAll(".nav-link");
    links.forEach(link => {
        link.addEventListener("click", (e) => {
            links.forEach(l => l.classList.remove("active"));
            link.classList.add("active");
        });
    });

    // Scroll spy
    const sections = document.querySelectorAll("section[id]");
    window.addEventListener("scroll", () => {
        const scrollY = window.scrollY + 100;
        sections.forEach(section => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute("id");
            if (scrollY >= top && scrollY < top + height) {
                links.forEach(l => l.classList.remove("active"));
                const active = document.querySelector(`.nav-link[data-section="${id}"]`);
                if (active) active.classList.add("active");
            }
        });
    });
}

// ── Load Stats & KPIs ────────────────────────────────────────
async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        statsData = data;

        // KPI Counters
        animateCounter("kpiTotalVal", data.dataset.total_customers, "", ",");
        animateCounter("kpiChurnVal", data.dataset.churn_rate, "%");
        animateCounter("kpiChargesVal", data.dataset.avg_monthly_charges, "", "$", true);
        animateCounter("kpiAtRiskVal", data.dataset.churned, "", ",");

        // Hero badges
        document.getElementById("heroModelName").textContent = data.model.best_model;
        document.getElementById("heroDataSize").textContent =
            data.dataset.total_customers.toLocaleString() + " Records";

        const bestModel = data.model[data.model.best_model];
        if (bestModel) {
            document.getElementById("heroAccuracy").textContent =
                (bestModel.roc_auc * 100).toFixed(1) + "% AUC";
        }

        // Charts
        renderChurnDistChart(data.dataset);
        renderContractChart(data.dataset);
        renderTenureChart(data.dataset);
        renderPaymentChart(data.dataset);
        renderFeatureChart(data.model);
        renderChargesChart(data.dataset);
        renderReasonsChart(data.dataset);
        renderModelCards(data.model);

    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

// ── Animated Counter ─────────────────────────────────────────
function animateCounter(elementId, target, suffix = "", prefix = "", isDecimal = false) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const duration = 2000;
    const start = performance.now();
    const startVal = 0;

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = startVal + (target - startVal) * eased;

        let display;
        if (isDecimal) {
            display = current.toFixed(2);
        } else if (suffix === "%") {
            display = current.toFixed(1);
        } else {
            display = Math.round(current).toLocaleString();
        }

        if (prefix === "$") display = "$" + display;
        if (suffix) display += suffix;

        el.textContent = display;

        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

// ── Chart.js Configuration ───────────────────────────────────
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: "#8b9dc3",
                font: { family: "Inter", size: 12 },
                padding: 16,
            }
        },
        tooltip: {
            backgroundColor: "rgba(12, 18, 32, 0.96)",
            titleColor: "#eef2f7",
            bodyColor: "#8b9dc3",
            borderColor: "rgba(99, 102, 241, 0.12)",
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            titleFont: { family: "Outfit", weight: "bold" },
            bodyFont: { family: "Inter" },
        }
    },
    scales: {
        x: {
            ticks: { color: "#5a6f94", font: { family: "Inter", size: 11 } },
            grid: { color: "rgba(99, 102, 241, 0.04)" },
        },
        y: {
            ticks: { color: "#5a6f94", font: { family: "Inter", size: 11 } },
            grid: { color: "rgba(99, 102, 241, 0.04)" },
        }
    }
};

function getGradient(ctx, c1, c2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, c1);
    gradient.addColorStop(1, c2);
    return gradient;
}

// ── Charts ───────────────────────────────────────────────────
function renderChurnDistChart(ds) {
    const ctx = document.getElementById("churnDistChart").getContext("2d");
    new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Active Customers", "Churned Customers"],
            datasets: [{
                data: [ds.not_churned, ds.churned],
                backgroundColor: ["rgba(16, 185, 129, 0.8)", "rgba(239, 68, 68, 0.8)"],
                borderColor: ["rgba(16, 185, 129, 1)", "rgba(239, 68, 68, 1)"],
                borderWidth: 2,
                hoverOffset: 10,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "70%",
            plugins: {
                legend: chartDefaults.plugins.legend,
                tooltip: chartDefaults.plugins.tooltip,
            }
        }
    });
}

function renderContractChart(ds) {
    const ctx = document.getElementById("contractChart").getContext("2d");
    const labels = Object.keys(ds.churn_by_contract);
    const values = Object.values(ds.churn_by_contract);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Churn Rate (%)",
                data: values,
                backgroundColor: [
                    "rgba(239, 68, 68, 0.7)",
                    "rgba(245, 158, 11, 0.7)",
                    "rgba(16, 185, 129, 0.7)"
                ],
                borderColor: [
                    "rgba(239, 68, 68, 1)",
                    "rgba(245, 158, 11, 1)",
                    "rgba(16, 185, 129, 1)"
                ],
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.6,
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
            },
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: v => v + "%",
                    }
                }
            }
        }
    });
}

function renderTenureChart(ds) {
    const ctx = document.getElementById("tenureChart").getContext("2d");
    const labels = Object.keys(ds.churn_by_tenure);
    const values = Object.values(ds.churn_by_tenure);

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Churn Rate (%)",
                data: values,
                borderColor: "rgba(99, 102, 241, 1)",
                backgroundColor: "rgba(99, 102, 241, 0.1)",
                fill: true,
                tension: 0.4,
                pointBackgroundColor: "rgba(99, 102, 241, 1)",
                pointBorderColor: "#fff",
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 8,
                borderWidth: 3,
            }]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
            },
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    beginAtZero: true,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: v => v + "%",
                    }
                }
            }
        }
    });
}

function renderPaymentChart(ds) {
    const ctx = document.getElementById("paymentChart").getContext("2d");
    const labels = Object.keys(ds.churn_by_payment).map(l =>
        l.length > 20 ? l.substring(0, 18) + "..." : l
    );
    const values = Object.values(ds.churn_by_payment);
    const colors = [
        "rgba(239, 68, 68, 0.7)", "rgba(6, 182, 212, 0.7)",
        "rgba(245, 158, 11, 0.7)", "rgba(16, 185, 129, 0.7)"
    ];
    const borders = [
        "rgba(239, 68, 68, 1)", "rgba(6, 182, 212, 1)",
        "rgba(245, 158, 11, 1)", "rgba(16, 185, 129, 1)"
    ];

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Churn Rate (%)",
                data: values,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.6,
            }]
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    beginAtZero: true,
                    max: 50,
                    ticks: {
                        ...chartDefaults.scales.x.ticks,
                        callback: v => v + "%",
                    }
                },
                y: chartDefaults.scales.y,
            }
        }
    });
}

function renderFeatureChart(model) {
    const ctx = document.getElementById("featureChart").getContext("2d");
    const features = model.feature_importances || [];
    const top = features.slice(0, 15);
    const labels = top.map(f => f.feature.replace(/_/g, " "));
    const values = top.map(f => f.importance);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Importance",
                data: values,
                backgroundColor: values.map((_, i) => {
                    const ratio = i / values.length;
                    return `rgba(${99 - ratio * 60}, ${102 + ratio * 80}, ${241 - ratio * 50}, 0.7)`;
                }),
                borderColor: values.map((_, i) => {
                    const ratio = i / values.length;
                    return `rgba(${99 - ratio * 60}, ${102 + ratio * 80}, ${241 - ratio * 50}, 1)`;
                }),
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.7,
            }]
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    beginAtZero: true,
                },
                y: {
                    ...chartDefaults.scales.y,
                    ticks: { color: "#8b9dc3", font: { family: "Inter", size: 11 } },
                }
            }
        }
    });
}

function renderChargesChart(ds) {
    const ctx = document.getElementById("chargesChart").getContext("2d");
    const distribution = ds.charges_distribution || {};
    const labels = Object.keys(distribution);
    const totals = labels.map(l => distribution[l].total);
    const churned = labels.map(l => distribution[l].churned);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Total",
                    data: totals,
                    backgroundColor: "rgba(99, 102, 241, 0.6)",
                    borderColor: "rgba(99, 102, 241, 1)",
                    borderWidth: 2,
                    borderRadius: 6,
                },
                {
                    label: "Churned",
                    data: churned,
                    backgroundColor: "rgba(239, 68, 68, 0.6)",
                    borderColor: "rgba(239, 68, 68, 1)",
                    borderWidth: 2,
                    borderRadius: 6,
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
            },
            scales: chartDefaults.scales,
        }
    });
}

function renderReasonsChart(ds) {
    const ctx = document.getElementById("reasonsChart").getContext("2d");
    const reasons = ds.top_churn_reasons || {};
    const labels = Object.keys(reasons).slice(0, 8).map(l =>
        l.length > 25 ? l.substring(0, 23) + "..." : l
    );
    const values = Object.values(reasons).slice(0, 8);

    const colors = [
        "rgba(239, 68, 68, 0.7)", "rgba(244, 63, 94, 0.7)",
        "rgba(236, 72, 153, 0.7)", "rgba(168, 85, 247, 0.7)",
        "rgba(139, 92, 246, 0.7)", "rgba(99, 102, 241, 0.7)",
        "rgba(59, 130, 246, 0.7)", "rgba(6, 182, 212, 0.7)",
    ];

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Count",
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace("0.7", "1")),
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.7,
            }]
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
            },
            scales: {
                x: { ...chartDefaults.scales.x, beginAtZero: true },
                y: {
                    ...chartDefaults.scales.y,
                    ticks: { color: "#94a3b8", font: { family: "Inter", size: 10 } },
                }
            }
        }
    });
}

// ── Model Performance Cards ──────────────────────────────────
function renderModelCards(model) {
    const container = document.getElementById("modelCards");
    if (!container) return;

    const modelNames = Object.keys(model).filter(k =>
        k !== "best_model" && k !== "feature_importances"
    );

    container.innerHTML = modelNames.map(name => {
        const m = model[name];
        const isBest = name === model.best_model;
        return `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-name">${name}</span>
                    ${isBest ? '<span class="model-card-best">Best Model</span>' : ''}
                </div>
                <div class="model-metrics-grid">
                    <div class="metric-item">
                        <span class="metric-val">${(m.accuracy * 100).toFixed(1)}%</span>
                        <span class="metric-name">Accuracy</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-val">${(m.precision * 100).toFixed(1)}%</span>
                        <span class="metric-name">Precision</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-val">${(m.recall * 100).toFixed(1)}%</span>
                        <span class="metric-name">Recall</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-val">${(m.f1 * 100).toFixed(1)}%</span>
                        <span class="metric-name">F1 Score</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-val">${(m.roc_auc * 100).toFixed(1)}%</span>
                        <span class="metric-name">ROC AUC</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-val">${(m.cv_f1_mean * 100).toFixed(1)}%</span>
                        <span class="metric-name">CV F1 Mean</span>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

// ── Prediction Form ──────────────────────────────────────────
function setupPredictionForm() {
    const form = document.getElementById("predictForm");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const btn = document.getElementById("btnPredict");
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
        btn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => { data[key] = value; });

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            const result = await res.json();

            if (result.error) {
                alert("Prediction error: " + result.error);
                return;
            }

            renderPredictionResult(result);

        } catch (err) {
            alert("Failed to get prediction: " + err.message);
        } finally {
            btn.innerHTML = '<i class="fas fa-bolt"></i> Predict Churn';
            btn.disabled = false;
        }
    });
}

function renderPredictionResult(result) {
    const container = document.getElementById("predictionResults");
    container.style.display = "block";
    container.scrollIntoView({ behavior: "smooth", block: "start" });

    // Badge
    const badge = document.getElementById("resultBadge");
    badge.textContent = result.prediction;
    badge.className = "result-prediction-badge " +
        (result.churn_probability >= 50 ? "churn" : "stay");

    // Gauge
    const prob = result.churn_probability;
    const gaugeColor = result.risk_color;
    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeValue = document.getElementById("gaugeValue");

    gaugeFill.style.setProperty("--gauge-percent", prob + "%");
    gaugeFill.style.setProperty("--gauge-color", gaugeColor);
    gaugeValue.textContent = prob + "%";
    gaugeValue.style.color = gaugeColor;

    // Risk level
    const riskBadge = document.getElementById("riskBadge");
    riskBadge.textContent = result.risk_level;
    riskBadge.style.background = gaugeColor + "22";
    riskBadge.style.color = gaugeColor;
    riskBadge.style.border = "1px solid " + gaugeColor + "44";

    document.getElementById("riskAction").textContent = result.risk_action;

    // Factors
    const factorsList = document.getElementById("factorsList");
    factorsList.innerHTML = result.contributing_factors.map(f => `
        <div class="factor-item">
            <span class="factor-impact ${f.impact}">${f.impact}</span>
            <div class="factor-text">
                <span class="factor-name">${f.factor}</span>
                <span class="factor-desc">${f.description}</span>
            </div>
        </div>
    `).join("");

    // Strategies
    const strategiesList = document.getElementById("strategiesList");
    strategiesList.innerHTML = result.retention_strategies.map(s => `
        <div class="strategy-card ${s.priority}">
            <div class="strategy-header">
                <div class="strategy-icon"><i class="fas fa-${s.icon}"></i></div>
                <span class="strategy-title">${s.title}</span>
                <span class="strategy-priority ${s.priority}">${s.priority}</span>
            </div>
            <p class="strategy-desc">${s.description}</p>
        </div>
    `).join("");
}

// ── Customer Explorer ────────────────────────────────────────
async function loadCustomers(page = 1) {
    currentPage = page;
    const search = document.getElementById("customerSearch").value;
    const sortBy = document.getElementById("sortBy").value;
    const sortDir = document.getElementById("sortDir").value;

    try {
        const params = new URLSearchParams({
            page, per_page: 15, search, sort_by: sortBy, sort_dir: sortDir,
        });

        const res = await fetch("/api/customers?" + params);
        const data = await res.json();

        renderCustomerTable(data.customers);
        renderPagination(data.page, data.total_pages);

    } catch (err) {
        console.error("Failed to load customers:", err);
    }
}

function renderCustomerTable(customers) {
    const tbody = document.getElementById("customersBody");
    tbody.innerHTML = customers.map(c => {
        const score = c.churn_score;
        const scoreColor = score >= 75 ? "#ef4444" :
            score >= 50 ? "#f59e0b" :
                score >= 25 ? "#eab308" : "#10b981";
        const isChurned = c.churn_label === "Yes";

        return `
            <tr>
                <td><strong>${c.customer_id}</strong></td>
                <td>${c.gender}</td>
                <td>${c.tenure_months} mo</td>
                <td>${c.contract}</td>
                <td>$${c.monthly_charges.toFixed(2)}</td>
                <td>${c.internet_service}</td>
                <td>
                    ${score}
                    <div class="churn-score-bar">
                        <div class="churn-score-fill" style="width:${score}%; background:${scoreColor};"></div>
                    </div>
                </td>
                <td>
                    <span class="status-badge ${isChurned ? 'churned' : 'active'}">
                        ${isChurned ? 'Churned' : 'Active'}
                    </span>
                </td>
                <td>
                    <button class="btn-view" onclick="viewCustomer('${c.customer_id}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    }).join("");
}

function renderPagination(currentPg, totalPages) {
    const container = document.getElementById("pagination");
    if (totalPages <= 1) { container.innerHTML = ""; return; }

    let html = "";

    html += `<button class="page-btn" onclick="loadCustomers(1)" ${currentPg === 1 ? 'disabled' : ''}>
        <i class="fas fa-angle-double-left"></i>
    </button>`;
    html += `<button class="page-btn" onclick="loadCustomers(${currentPg - 1})" ${currentPg === 1 ? 'disabled' : ''}>
        <i class="fas fa-angle-left"></i>
    </button>`;

    const startPg = Math.max(1, currentPg - 2);
    const endPg = Math.min(totalPages, currentPg + 2);

    for (let i = startPg; i <= endPg; i++) {
        html += `<button class="page-btn ${i === currentPg ? 'active' : ''}" onclick="loadCustomers(${i})">${i}</button>`;
    }

    html += `<button class="page-btn" onclick="loadCustomers(${currentPg + 1})" ${currentPg === totalPages ? 'disabled' : ''}>
        <i class="fas fa-angle-right"></i>
    </button>`;
    html += `<button class="page-btn" onclick="loadCustomers(${totalPages})" ${currentPg === totalPages ? 'disabled' : ''}>
        <i class="fas fa-angle-double-right"></i>
    </button>`;

    container.innerHTML = html;
}

function setupCustomerSearch() {
    const input = document.getElementById("customerSearch");
    input.addEventListener("input", () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => loadCustomers(1), 400);
    });

    document.getElementById("sortBy").addEventListener("change", () => loadCustomers(1));
    document.getElementById("sortDir").addEventListener("change", () => loadCustomers(1));
}

// ── Customer Modal ───────────────────────────────────────────
function setupModal() {
    document.getElementById("modalClose").addEventListener("click", closeModal);
    document.getElementById("customerModal").addEventListener("click", (e) => {
        if (e.target === e.currentTarget) closeModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeModal();
    });
}

function closeModal() {
    document.getElementById("customerModal").style.display = "none";
    document.body.style.overflow = "";
}

async function viewCustomer(customerId) {
    const modal = document.getElementById("customerModal");
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";

    document.getElementById("modalCustomerId").textContent = customerId;
    document.getElementById("modalBody").innerHTML =
        '<div class="modal-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const res = await fetch(`/api/customer/${customerId}`);
        const data = await res.json();

        if (data.error) {
            document.getElementById("modalBody").innerHTML =
                `<div class="modal-loading" style="color: #f87171;">${data.error}</div>`;
            return;
        }

        renderCustomerModal(data);

    } catch (err) {
        document.getElementById("modalBody").innerHTML =
            `<div class="modal-loading" style="color: #f87171;">Error loading customer data</div>`;
    }
}

function renderCustomerModal(data) {
    const c = data.customer;
    const p = data.prediction;

    const riskColor = p.risk_color;
    const probColor = p.churn_probability >= 50 ? "#ef4444" : "#10b981";

    let html = `
        <div class="modal-risk-header">
            <span class="modal-risk-prob" style="color:${probColor}">${p.churn_probability}%</span>
            <span class="modal-risk-label" style="background:${riskColor}22; color:${riskColor}; border:1px solid ${riskColor}44;">
                ${p.risk_level} Risk
            </span>
        </div>
        <p style="text-align:center; color:var(--text-secondary); margin-bottom:24px;">${p.risk_action}</p>

        <div class="modal-info-grid">
            <div class="modal-info-item">
                <span class="info-label">Gender</span>
                <span class="info-value">${c.gender}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Tenure</span>
                <span class="info-value">${c.tenure_months} months</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Contract</span>
                <span class="info-value">${c.contract}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Monthly Charges</span>
                <span class="info-value">$${parseFloat(c.monthly_charges).toFixed(2)}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Internet Service</span>
                <span class="info-value">${c.internet_service}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Payment Method</span>
                <span class="info-value">${c.payment_method}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">City</span>
                <span class="info-value">${c.city || 'N/A'}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">CLTV</span>
                <span class="info-value">$${c.cltv}</span>
            </div>
            <div class="modal-info-item">
                <span class="info-label">Actual Status</span>
                <span class="info-value" style="color:${c.churn_label === 'Yes' ? '#f87171' : '#34d399'}">
                    ${c.churn_label === 'Yes' ? 'Churned' : 'Active'}
                </span>
            </div>
        </div>
    `;

    if (c.churn_reason && c.churn_label === "Yes") {
        html += `
            <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.2); border-radius:8px; padding:14px 18px; margin-bottom:24px;">
                <strong style="color:#f87171;"><i class="fas fa-exclamation-circle"></i> Churn Reason:</strong>
                <span style="color:var(--text-secondary); margin-left:8px;">${c.churn_reason}</span>
            </div>
        `;
    }

    // Factors
    if (p.contributing_factors && p.contributing_factors.length > 0) {
        html += `<h4 style="margin-bottom:12px;"><i class="fas fa-search-plus" style="color:#3b82f6;margin-right:8px;"></i>Contributing Factors</h4>`;
        html += `<div class="factors-list" style="margin-bottom:24px;">`;
        p.contributing_factors.forEach(f => {
            html += `
                <div class="factor-item">
                    <span class="factor-impact ${f.impact}">${f.impact}</span>
                    <div class="factor-text">
                        <span class="factor-name">${f.factor}</span>
                        <span class="factor-desc">${f.description}</span>
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    }

    // Strategies
    if (p.retention_strategies && p.retention_strategies.length > 0) {
        html += `<h4 style="margin-bottom:12px;"><i class="fas fa-lightbulb" style="color:#f59e0b;margin-right:8px;"></i>Retention Strategies</h4>`;
        html += `<div class="strategies-list">`;
        p.retention_strategies.forEach(s => {
            html += `
                <div class="strategy-card ${s.priority}">
                    <div class="strategy-header">
                        <div class="strategy-icon"><i class="fas fa-${s.icon}"></i></div>
                        <span class="strategy-title">${s.title}</span>
                        <span class="strategy-priority ${s.priority}">${s.priority}</span>
                    </div>
                    <p class="strategy-desc">${s.description}</p>
                </div>
            `;
        });
        html += `</div>`;
    }

    document.getElementById("modalBody").innerHTML = html;
}
