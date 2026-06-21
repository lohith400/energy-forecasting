/**
 * E-City — Application Logic
 * Powering India's Grid Intelligence
 *
 * Features:
 *  1. SVG India Map region selection with hover & click interactions.
 *  2. Typewriter animated terminal for AI-generated insight messages.
 *  3. Single-axis Chart.js line chart (demand only — no secondary temp axis).
 *  4. Regional bar comparison chart.
 *  5. Full backend integration with /predict, /get_history, /get_temperature.
 *  6. Recursive LSTM 14-feature shape contract honored.
 */

(function () {
  'use strict';

  // ── Constants ────────────────────────────────────────────────────────────────
  const API_BASE    = 'http://localhost:5000';
  const DATASET_END = new Date('2024-04-30');
  const MONTHS      = ['January','February','March','April','May','June','July',
                       'August','September','October','November','December'];
  const DAYS        = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];

  const REGION_LABELS = {
    North:     'Northern Grid',
    South:     'Southern Grid',
    East:      'Eastern Grid',
    West:      'Western Grid',
    NorthEast: 'North-Eastern Grid',
  };

  const REGION_CITIES = {
    North:     'New Delhi',
    South:     'Bengaluru',
    East:      'Kolkata',
    West:      'Mumbai',
    NorthEast: 'Guwahati',
  };

  const REGION_COLORS = {
    North:     '#2563EB',
    South:     '#059669',
    East:      '#D97706',
    West:      '#7C3AED',
    NorthEast: '#0891B2',
  };

  const REGION_DOT_COLORS = {
    North:     '#2563EB',
    South:     '#059669',
    East:      '#D97706',
    West:      '#7C3AED',
    NorthEast: '#0891B2',
  };

  const SEASONAL_AVG = {
    North: 52, South: 42, East: 28, West: 50, NorthEast: 8,
  };

  // ── State ────────────────────────────────────────────────────────────────────
  let state = {
    region:          null,
    date:            null,
    dateObj:         null,
    month:           null,
    dayOfWeek:       null,
    hour:            12,
    temperature:     null,
    tempAutoFetched: false,
    isHoliday:       false,
    isHistorical:    false,
    lastPrediction:  null,
    regionalHourlyTemperatures: [],
    nationalHourlyTemperatures: [],
  };

  let hourlyChart   = null;
  let regionalChart = null;

  // Typewriter state
  let typewriterQueue   = [];
  let typewriterActive  = false;
  let typewriterTimeout = null;

  // ── DOM references ───────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const dom = {
    headerTime:          $('header-time'),
    headerDate:          $('header-date'),
    forecastDate:        $('forecast-date'),
    chipMonth:           $('chip-month'),
    chipMonthText:       $('chip-month-text'),
    chipDay:             $('chip-day'),
    chipDayText:         $('chip-day-text'),
    modeIndicator:       $('mode-indicator'),
    hourSlider:          $('hour-slider'),
    hourDisplay:         $('hour-display'),
    hourDisplayText:     $('hour-display-text'),
    tempInput:           $('temp-input'),
    tempStatus:          $('temp-status'),
    tempStatusText:      $('temp-status-text'),
    holidayCheckbox:     $('holiday-checkbox'),
    holidayWarning:      $('holiday-warning'),
    forecastBtn:         $('forecast-btn'),
    progressWrapper:     $('progress-wrapper'),
    progressBar:         $('progress-bar'),
    resultsSection:      $('results-section'),
    predictionValue:     $('prediction-value'),
    predictionContext:   $('prediction-context'),
    predictionRangeText: $('prediction-range-text'),
    statusBadge:         $('status-badge'),
    statusBadgeText:     $('status-badge-text'),
    hourlyChartCanvas:   $('hourly-chart'),
    regionalChartCanvas: $('regional-chart'),
    insightPeak:         $('insight-peak'),
    insightTemp:         $('insight-temp'),
    insightYoy:          $('insight-yoy'),
    chartModeBadge:      $('chart-mode-badge'),
    legendActual:        $('legend-actual'),
    terminalBody:        $('terminal-body'),
    // Map elements
    indiaMap:            $('india-map'),
    mapTooltip:          $('map-tooltip'),
    selectedRegionText:  $('selected-region-text'),
    selectedRegionDot:   $('selected-region-dot'),
    selectedRegionDisplay: $('selected-region-display'),
  };

  // ── Clock ────────────────────────────────────────────────────────────────────
  function updateClock() {
    const now = new Date();
    const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const hh  = String(ist.getHours()).padStart(2, '0');
    const mm  = String(ist.getMinutes()).padStart(2, '0');
    const ss  = String(ist.getSeconds()).padStart(2, '0');
    if (dom.headerTime) dom.headerTime.textContent = `${hh}:${mm}:${ss}`;
    if (dom.headerDate) dom.headerDate.textContent =
      ist.toLocaleDateString('en-IN', { weekday:'short', month:'short', day:'numeric', year:'numeric' });
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ── SVG Map Interactions ─────────────────────────────────────────────────────
  function initMap() {
    const regions = dom.indiaMap.querySelectorAll('.map-region');
    regions.forEach((regionEl) => {
      const regionId = regionEl.dataset.region;

      regionEl.addEventListener('mouseenter', (e) => {
        const tooltip = dom.mapTooltip;
        const label   = REGION_LABELS[regionId] || regionId;
        const city    = REGION_CITIES[regionId] || '';
        tooltip.innerHTML = `<strong>${label}</strong>${city ? '<br>' + city : ''}`;
        tooltip.classList.add('visible');
        positionTooltip(e);
      });

      regionEl.addEventListener('mousemove', positionTooltip);

      regionEl.addEventListener('mouseleave', () => {
        dom.mapTooltip.classList.remove('visible');
      });

      regionEl.addEventListener('click', () => {
        selectRegion(regionId);
      });
    });
  }

  function positionTooltip(e) {
    const rect    = dom.indiaMap.closest('.map-container').getBoundingClientRect();
    const tooltip = dom.mapTooltip;
    const x       = e.clientX - rect.left + 10;
    const y       = e.clientY - rect.top  - 10;
    tooltip.style.left = Math.min(x, rect.width - 160) + 'px';
    tooltip.style.top  = Math.max(y - 40, 4) + 'px';
  }

  function selectRegion(regionId) {
    state.region = regionId;

    // Clear all region selected classes
    dom.indiaMap.querySelectorAll('.map-region').forEach((el) => {
      el.classList.remove('selected-north','selected-south','selected-east',
                          'selected-west','selected-northeast');
    });

    // Apply selected class
    const el = dom.indiaMap.querySelector(`[data-region="${regionId}"]`);
    if (el) {
      const cls = 'selected-' + regionId.toLowerCase().replace('northeast','northeast');
      el.classList.add(cls);
    }

    // Update display pill
    const label = REGION_LABELS[regionId] || regionId;
    const color = REGION_DOT_COLORS[regionId] || '#94A3B8';
    dom.selectedRegionText.textContent = label;
    dom.selectedRegionDot.style.background = color;

    // Fetch temperature for current date/region if date is set
    if (state.date) fetchTemperature();
  }

  // ── Hour Slider ──────────────────────────────────────────────────────────────
  function formatHour(h) {
    if (h === 0)  return '12:00 AM';
    if (h === 12) return '12:00 PM';
    return h < 12 ? `${h}:00 AM` : `${h - 12}:00 PM`;
  }

  function updateHourSlider() {
    const h   = parseInt(dom.hourSlider.value, 10);
    state.hour = h;
    dom.hourDisplayText.textContent = formatHour(h);
    const pct = (h / 23) * 100;
    dom.hourSlider.style.background =
      `linear-gradient(to right, #2563EB ${pct}%, #E2E8F0 ${pct}%)`;
  }

  dom.hourSlider.addEventListener('input', updateHourSlider);
  updateHourSlider();

  // ── Date Picker ──────────────────────────────────────────────────────────────
  function updateDateInfo(dateStr) {
    if (!dateStr) return;
    const d         = new Date(dateStr + 'T00:00:00');
    state.dateObj   = d;
    state.month     = d.getMonth();
    state.dayOfWeek = d.getDay();
    state.date      = dateStr;

    dom.chipMonthText.textContent = MONTHS[d.getMonth()];
    dom.chipDayText.textContent   = DAYS[d.getDay()];

    const isHistorical = d <= DATASET_END;
    state.isHistorical  = isHistorical;

    if (dom.modeIndicator) {
      dom.modeIndicator.style.display = 'flex';
      if (isHistorical) {
        dom.modeIndicator.className  = 'mode-indicator historical';
        dom.modeIndicator.innerHTML  =
          '<svg viewBox="0 0 24 24" fill="currentColor" style="width:12px;height:12px;"><path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9z"/></svg>'
          + ' Historical Mode — Actual vs Predicted';
      } else {
        dom.modeIndicator.className  = 'mode-indicator future';
        dom.modeIndicator.innerHTML  =
          '<svg viewBox="0 0 24 24" fill="currentColor" style="width:12px;height:12px;"><path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z"/></svg>'
          + ' Forecast Mode — Future Prediction';
      }
    }

    if (state.region) fetchTemperature();
  }

  dom.forecastDate.addEventListener('change', () => {
    updateDateInfo(dom.forecastDate.value);
  });

  // ── Temperature Fetch ────────────────────────────────────────────────────────
  async function fetchTemperature() {
    if (!state.region || !state.date) return;
    try {
      const url  = `${API_BASE}/get_temperature?region=${state.region}&date=${state.date}&hour=${state.hour}`;
      const resp = await fetch(url);
      const data = await resp.json();
      if (data.temperature_celsius !== undefined) {
        const t                 = parseFloat(data.temperature_celsius);
        dom.tempInput.value     = t.toFixed(1);
        state.temperature       = t;
        state.tempAutoFetched   = true;
        state.regionalHourlyTemperatures = data.regional_hourly_temperatures || [];
        state.nationalHourlyTemperatures = data.national_hourly_temperatures || [];
        dom.tempStatus.className       = 'temp-status fetched';
        dom.tempStatusText.textContent = `Auto: ${data.city || state.region} ${data.source ? '(' + data.source + ')' : ''}`;
        dom.tempStatus.querySelector('svg').innerHTML = '<path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>';
      }
    } catch (_) {
      // Silently ignore — user can enter manually
    }
  }

  dom.tempInput.addEventListener('input', () => {
    const v = parseFloat(dom.tempInput.value);
    if (!isNaN(v)) {
      state.temperature     = v;
      state.tempAutoFetched = false;
      dom.tempStatus.className       = 'temp-status manual';
      dom.tempStatusText.textContent = 'Manual override';
    }
  });

  // ── Holiday Toggle ───────────────────────────────────────────────────────────
  dom.holidayCheckbox.addEventListener('change', () => {
    state.isHoliday = dom.holidayCheckbox.checked;
    dom.holidayWarning.classList.toggle('visible', state.isHoliday);
  });

  // ── Typewriter Engine ────────────────────────────────────────────────────────
  function clearTerminal() {
    if (!dom.terminalBody) return;
    dom.terminalBody.innerHTML = '';
    typewriterQueue  = [];
    typewriterActive = false;
    if (typewriterTimeout) { clearTimeout(typewriterTimeout); typewriterTimeout = null; }
  }

  function addTerminalLine(prompt, text, cssClass) {
    typewriterQueue.push({ prompt, text, cssClass });
    if (!typewriterActive) processTypewriterQueue();
  }

  function processTypewriterQueue() {
    if (typewriterQueue.length === 0) {
      typewriterActive = false;
      return;
    }
    typewriterActive = true;
    const item = typewriterQueue.shift();
    typewriteLine(item.prompt, item.text, item.cssClass, () => {
      typewriterTimeout = setTimeout(processTypewriterQueue, 120);
    });
  }

  function typewriteLine(prompt, text, cssClass, onDone) {
    const lineEl = document.createElement('div');
    lineEl.className = 'terminal-line';

    const promptEl = document.createElement('span');
    promptEl.className = 'terminal-prompt';
    promptEl.textContent = prompt;
    lineEl.appendChild(promptEl);

    const textEl = document.createElement('span');
    textEl.className = `terminal-output${cssClass ? ' ' + cssClass : ''}`;
    lineEl.appendChild(textEl);

    // Cursor
    const cursorEl = document.createElement('span');
    cursorEl.className = 'terminal-cursor';
    lineEl.appendChild(cursorEl);

    dom.terminalBody.appendChild(lineEl);
    dom.terminalBody.scrollTop = dom.terminalBody.scrollHeight;

    let i = 0;
    const speed = Math.max(18, Math.min(35, 900 / text.length));

    function type() {
      if (i < text.length) {
        textEl.textContent += text[i++];
        dom.terminalBody.scrollTop = dom.terminalBody.scrollHeight;
        typewriterTimeout = setTimeout(type, speed);
      } else {
        lineEl.removeChild(cursorEl);
        if (onDone) onDone();
      }
    }
    type();
  }

  function runInsightSequence(predData, region, date, hour, isHoliday) {
    clearTerminal();

    const regionLabel  = REGION_LABELS[region] || region;
    const city         = REGION_CITIES[region] || region;
    const demand       = predData.predicted_demand_gw;
    const national     = predData.national_hourly_gw;
    const hourlyFcst   = predData.hourly_forecast || [];
    const status       = predData.status || 'Normal Load';
    const isHistorical = predData.is_historical;

    // Peak hour computation
    let peakHour = 0;
    let peakVal  = 0;
    hourlyFcst.forEach((v, i) => { if (v > peakVal) { peakVal = v; peakHour = i; } });

    // Demand level classification
    const avg     = SEASONAL_AVG[region] || 30;
    const pctDiff = ((demand - avg) / avg * 100).toFixed(1);
    const isHigh  = demand > avg * 1.10;
    const isLow   = demand < avg * 0.90;

    // National total
    const natTotal = national && national.length
      ? (national[hour] || 0).toFixed(2)
      : '—';

    // Hour label
    const hourLabel = formatHour(hour);

    // Insight message from server or self-generated
    const serverMsg = predData.insight_message || null;

    addTerminalLine('ecity@grid:~$', ` initializing E-City insight engine...`, 'info');

    setTimeout(() => {
      addTerminalLine('>', ` [${new Date().toISOString().replace('T',' ').slice(0,19)}] Analysis started`, 'accent');
      addTerminalLine('>', ` Region: ${regionLabel} (${city}) | Mode: ${isHistorical ? 'Historical' : 'Forecast'}`, 'info');
      addTerminalLine('>', ` Selected hour: ${hourLabel} | Holiday override: ${isHoliday ? 'YES' : 'NO'}`, 'info');
    }, 300);

    setTimeout(() => {
      addTerminalLine('predict:$', ` LSTM inference complete — demand at ${hourLabel}`, '');
      addTerminalLine('result >', ` ${regionLabel}: ${demand} GW  |  National: ${natTotal} GW`, 'success');
      addTerminalLine('result >', ` Confidence band: [${predData.confidence_low} – ${predData.confidence_high}] GW`, '');
      addTerminalLine('status >', ` Grid status: ${status}`, status.includes('Alert') ? 'warning' : status.includes('Warning') ? 'warning' : 'success');
    }, 800);

    setTimeout(() => {
      addTerminalLine('insight:$', ` Running demand pattern analysis...`, 'info');

      // Dynamic insight logic
      if (serverMsg) {
        addTerminalLine('💡 AI >', ` ${serverMsg}`, 'success');
      } else {
        if (isHoliday) {
          addTerminalLine('💡 AI >', ` Holiday detected: grid load is expected ~15–20% below weekday baseline.`, 'warning');
          addTerminalLine('💡 AI >', ` Residential demand remains steady; industrial & commercial sectors show sharp drop.`, '');
        } else if (hour >= 18 && hour <= 22) {
          addTerminalLine('💡 AI >', ` Evening peak window (18:00–22:00 IST) — highest demand corridor of the day.`, 'warning');
          addTerminalLine('💡 AI >', ` Recommend grid operators pre-activate peaker plants 30 min before ramp-up.`, '');
        } else if (hour >= 2 && hour <= 5) {
          addTerminalLine('💡 AI >', ` Off-peak trough window. Optimal for grid maintenance and energy storage charging.`, 'success');
        } else if (hour >= 9 && hour <= 12) {
          addTerminalLine('💡 AI >', ` Morning ramp period — solar generation supplementing grid; demand rising steadily.`, 'info');
        } else {
          addTerminalLine('💡 AI >', ` Standard demand window. Grid conditions nominal across the ${regionLabel}.`, 'success');
        }

        if (isHigh) {
          addTerminalLine('⚠️  ALERT>', ` Demand (${demand} GW) is ${pctDiff}% above seasonal average (${avg} GW). High load risk.`, 'warning');
        } else if (isLow) {
          addTerminalLine('📉 NOTE >', ` Demand (${demand} GW) is ${Math.abs(parseFloat(pctDiff))}% below seasonal average. Surplus capacity available.`, 'info');
        } else {
          addTerminalLine('✅ NOMINAL>', ` Demand within ±10% of seasonal average. Grid balanced.`, 'success');
        }

        addTerminalLine('💡 AI >', ` Peak hour forecast: ${formatHour(peakHour)} at ${peakVal.toFixed(2)} GW — plan dispatch accordingly.`, 'accent');
      }

      addTerminalLine('ecity@grid:~$', ` analysis complete. ✓`, 'success');
    }, 1400);
  }

  // ── Progress Bar ─────────────────────────────────────────────────────────────
  function startProgress() {
    dom.progressWrapper.style.display = 'block';
    dom.progressBar.style.width = '0%';
    setTimeout(() => { dom.progressBar.style.width = '40%'; }, 50);
    setTimeout(() => { dom.progressBar.style.width = '70%'; }, 400);
    setTimeout(() => { dom.progressBar.style.width = '88%'; }, 900);
  }

  function completeProgress() {
    dom.progressBar.style.width = '100%';
    setTimeout(() => { dom.progressWrapper.style.display = 'none'; dom.progressBar.style.width = '0%'; }, 500);
  }

  // ── Chart Rendering ──────────────────────────────────────────────────────────
  const HOURS_24 = Array.from({ length: 24 }, (_, i) => formatHour(i));

  function renderHourlyChart(forecast, actual, selectedHour) {
    if (hourlyChart) { hourlyChart.destroy(); hourlyChart = null; }

    const datasets = [{
      label:           'Predicted (GW)',
      data:            forecast,
      borderColor:     '#2563EB',
      backgroundColor: 'rgba(37,99,235,0.07)',
      borderWidth:     2.5,
      tension:         0.4,
      fill:            true,
      pointRadius:     forecast.map((_, i) => i === selectedHour ? 7 : 3),
      pointBackgroundColor: forecast.map((_, i) =>
        i === selectedHour ? '#fff' : '#2563EB'),
      pointBorderColor: '#2563EB',
      pointBorderWidth: forecast.map((_, i) => i === selectedHour ? 3 : 1),
    }];

    if (actual && actual.length === 24) {
      datasets.push({
        label:           'Actual (GW)',
        data:            actual,
        borderColor:     '#e65100',
        backgroundColor: 'rgba(230,81,0,0.05)',
        borderWidth:     2,
        tension:         0.4,
        fill:            false,
        pointRadius:     3,
        pointBackgroundColor: '#e65100',
        borderDash:      [5, 3],
      });
      dom.legendActual.style.display = 'flex';
    } else {
      dom.legendActual.style.display = 'none';
    }

    hourlyChart = new Chart(dom.hourlyChartCanvas, {
      type: 'line',
      data: { labels: HOURS_24, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1E293B',
            titleColor: '#fff',
            bodyColor: '#CBD5E1',
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} GW`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
            ticks: {
              font: { size: 11, family: "'Inter', sans-serif" },
              color: '#94A3B8',
              maxTicksLimit: 12,
              maxRotation: 0,
            },
          },
          y: {
            grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
            ticks: {
              font: { size: 11, family: "'Inter', sans-serif" },
              color: '#94A3B8',
              callback: (v) => v.toFixed(1) + ' GW',
            },
          },
        },
        animation: { duration: 600, easing: 'easeInOutQuart' },
      },
    });
  }

  function renderRegionalChart(regionalComparison, selectedRegion) {
    if (regionalChart) { regionalChart.destroy(); regionalChart = null; }

    const labels  = Object.keys(regionalComparison).map((r) => r.replace('NorthEast', 'N-East'));
    const rawKeys = Object.keys(regionalComparison);
    const values  = rawKeys.map((r) => regionalComparison[r]);
    const colors  = rawKeys.map((r) =>
      r === selectedRegion
        ? (REGION_COLORS[r] || '#2563EB')
        : (REGION_COLORS[r] + '88' || '#2563EB88')
    );

    regionalChart = new Chart(dom.regionalChartCanvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label:           'Demand (GW)',
          data:            values,
          backgroundColor: colors,
          borderRadius:    8,
          borderSkipped:   false,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1E293B',
            titleColor: '#fff',
            bodyColor: '#CBD5E1',
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              label: (ctx) => ` ${ctx.parsed.y.toFixed(2)} GW`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              font: { size: 11, family: "'Inter', sans-serif", weight: '600' },
              color: '#475569',
            },
          },
          y: {
            grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
            ticks: {
              font: { size: 11, family: "'Inter', sans-serif" },
              color: '#94A3B8',
              callback: (v) => v.toFixed(1) + ' GW',
            },
          },
        },
        animation: { duration: 600, easing: 'easeInOutQuart' },
      },
    });
  }

  // ── Populate Results ─────────────────────────────────────────────────────────
  function populatePredictionCard(data, region, hour) {
    const demand  = data.predicted_demand_gw;
    const label   = REGION_LABELS[region] || region;
    const hourStr = formatHour(hour);
    const dayStr  = state.dateObj
      ? state.dateObj.toLocaleDateString('en-IN', { weekday:'long', month:'long', day:'numeric' })
      : '';

    dom.predictionValue.textContent     = demand.toFixed(2);
    dom.predictionContext.textContent   = `${label} · ${hourStr}${dayStr ? ' · ' + dayStr : ''}`;
    dom.predictionRangeText.textContent =
      `Confidence: ${data.confidence_low} – ${data.confidence_high} GW`;

    const status  = data.status || 'Normal Load';
    const badge   = dom.statusBadge;
    const badgeClass = status.includes('Alert') ? 'alert'
                     : status.includes('Warning') ? 'warning'
                     : 'normal';
    badge.className            = `status-badge ${badgeClass}`;
    dom.statusBadgeText.textContent = status;

    const chartBadge = dom.chartModeBadge;
    if (data.is_historical) {
      chartBadge.textContent         = 'Historical';
      chartBadge.style.background    = '#EFF6FF';
      chartBadge.style.color         = '#2563EB';
    } else {
      chartBadge.textContent         = 'Forecast';
      chartBadge.style.background    = '#ECFDF5';
      chartBadge.style.color         = '#059669';
    }
  }

  function populateInsightStrip(data, region) {
    const hourlyFcst = data.hourly_forecast || [];
    // Peak hour
    let peakH = 0; let peakV = 0;
    hourlyFcst.forEach((v, i) => { if (v > peakV) { peakV = v; peakH = i; } });
    dom.insightPeak.textContent = peakV ? `${formatHour(peakH)} · ${peakV.toFixed(1)} GW` : '—';

    // Temp
    const temp = state.temperature;
    dom.insightTemp.textContent = temp !== null && temp !== undefined
      ? `${temp.toFixed(1)}°C`
      : '—';

    // YoY placeholder
    dom.insightYoy.textContent = data.is_historical ? '+2.3% YoY' : 'Live Forecast';
  }

  // ── Forecast Submission ──────────────────────────────────────────────────────
  dom.forecastBtn.addEventListener('click', async () => {
    // Validations
    if (!state.region) {
      alert('Please select a region on the India map first.');
      return;
    }
    if (!state.date) {
      alert('Please select a forecast date.');
      return;
    }
    if (state.temperature === null || state.temperature === undefined) {
      alert('Please provide a temperature value.');
      return;
    }

    dom.forecastBtn.disabled = true;
    dom.forecastBtn.textContent = 'Running LSTM Inference…';
    startProgress();
    clearTerminal();
    addTerminalLine('ecity@grid:~$', ' connecting to LSTM inference engine...', 'info');

    try {
      // Build national hourly temps array
      let nationalHourlyTemps = state.nationalHourlyTemperatures;
      if (!nationalHourlyTemps || nationalHourlyTemps.length !== 24) {
        nationalHourlyTemps = Array(24).fill(parseFloat(state.temperature));
      }

      const payload = {
        region:                      state.region,
        date:                        state.date,
        hour:                        state.hour,
        is_holiday:                  state.isHoliday,
        national_hourly_temperatures: nationalHourlyTemps,
      };

      const resp = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const predData = await resp.json();
      if (predData.error) throw new Error(predData.error);

      state.lastPrediction = predData;

      // Fetch actual if historical
      let actualData = null;
      if (predData.is_historical) {
        try {
          const histResp = await fetch(
            `${API_BASE}/get_history?date=${state.date}&region=${state.region}`
          );
          const histJson = await histResp.json();
          if (histJson.available) actualData = histJson.hourly_actual;
        } catch (_) {}
      }

      completeProgress();
      populatePredictionCard(predData, state.region, state.hour);
      populateInsightStrip(predData, state.region);
      renderHourlyChart(predData.hourly_forecast, actualData, state.hour);
      renderRegionalChart(predData.regional_comparison, state.region);
      runInsightSequence(predData, state.region, state.date, state.hour, state.isHoliday);

    } catch (err) {
      completeProgress();
      clearTerminal();
      addTerminalLine('ERROR >', ` ${err.message}`, 'warning');
      addTerminalLine('ecity@grid:~$', ' Is the backend server running on localhost:5000?', 'info');
      dom.predictionValue.textContent = '—';
      dom.predictionContext.textContent = 'Error — check console';
    } finally {
      dom.forecastBtn.disabled   = false;
      dom.forecastBtn.innerHTML =
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z"/></svg> Run E-City Forecast';
    }
  });

  // ── Auto-set today's date & default hour ────────────────────────────────────
  function setDefaultDate() {
    const today = new Date();
    const ist   = new Date(today.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const yyyy  = ist.getFullYear();
    const mm    = String(ist.getMonth() + 1).padStart(2, '0');
    const dd    = String(ist.getDate()).padStart(2, '0');
    const dateStr = `${yyyy}-${mm}-${dd}`;
    dom.forecastDate.value = dateStr;
    dom.hourSlider.value   = String(ist.getHours());
    updateHourSlider();
    updateDateInfo(dateStr);
  }

  // ── Init ──────────────────────────────────────────────────────────────────────
  initMap();
  setDefaultDate();

})();