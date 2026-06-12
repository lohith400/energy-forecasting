/**
 * GridSense India — Application Logic (Fixed)
 * Handles all interactivity, API calls, Chart.js rendering, and live clock.
 *
 * Key fixes:
 *  1. No min-date restriction — historical dates allowed from Jan 2019.
 *  2. Detects historical vs future dates (dataset ends Apr 30 2024).
 *  3. For historical dates: fetches actual demand via /get_history and renders
 *     both Actual and Predicted lines on the chart.
 *  4. Mode indicator shows "Historical Analysis" or "Future Forecast".
 *  5. Temperature auto-fetch falls back to monthly average for historical dates.
 */

(function () {
  'use strict';

  // ============================================
  // Constants
  // ============================================
  const API_BASE = 'http://localhost:5000';
  const DATASET_END = new Date('2024-04-30'); // last date in dataset
  const MONTHS = ['January','February','March','April','May','June','July',
                  'August','September','October','November','December'];
  const DAYS   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const REGION_LABELS = {
    North:     'Northern Grid',
    South:     'Southern Grid',
    East:      'Eastern Grid',
    West:      'Western Grid',
    NorthEast: 'North-Eastern Grid',
  };

  // Seasonal average baselines per region (GW) — for status badge comparison
  const SEASONAL_AVG = {
    North: 52, South: 42, East: 28, West: 50, NorthEast: 8,
  };

  // ============================================
  // State
  // ============================================
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

  // ============================================
  // DOM References
  // ============================================
  const $ = (id) => document.getElementById(id);

  const dom = {
    headerTime:          $('header-time'),
    headerDate:          $('header-date'),
    regionPills:         $('region-pills'),
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
    legendRegionalTemp:  $('legend-regional-temp'),
    legendNationalTemp:  $('legend-national-temp'),
  };

  // ============================================
  // Live IST Clock
  // ============================================
  function updateClock() {
    const now = new Date();
    const istOffset = 5.5 * 60 * 60 * 1000;
    const utc = now.getTime() + now.getTimezoneOffset() * 60000;
    const ist = new Date(utc + istOffset);

    const hours = ist.getHours();
    const mins  = ist.getMinutes();
    const secs  = ist.getSeconds();
    const ampm  = hours >= 12 ? 'PM' : 'AM';
    const h12   = hours % 12 || 12;

    dom.headerTime.textContent =
      `${h12}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')} ${ampm}`;
    dom.headerDate.textContent =
      `${DAYS[ist.getDay()]}, ${ist.getDate()} ${MONTHS[ist.getMonth()]} ${ist.getFullYear()}`;
  }

  // ============================================
  // Region Selection
  // ============================================
  function initRegionPills() {
    const pills = dom.regionPills.querySelectorAll('.region-pill');
    pills.forEach((pill) => {
      pill.addEventListener('click', () => {
        pills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        state.region = pill.dataset.region;
        tryFetchTemperature();
      });
    });
  }

  // ============================================
  // Date Picker  (no min restriction — allows 2019+)
  // ============================================
  function isHistoricalDate(dateObj) {
    // A date is "historical" if it's on or before the last dataset date
    const d = new Date(dateObj.getFullYear(), dateObj.getMonth(), dateObj.getDate());
    const end = new Date(DATASET_END.getFullYear(), DATASET_END.getMonth(), DATASET_END.getDate());
    return d <= end;
  }

  function updateModeIndicator() {
    if (!state.date) {
      dom.modeIndicator.style.display = 'none';
      return;
    }
    dom.modeIndicator.style.display = 'block';
    if (state.isHistorical) {
      dom.modeIndicator.style.cssText = [
        'display:block',
        'margin-top:6px',
        'padding:4px 12px',
        'border-radius:20px',
        'font-size:11px',
        'font-weight:700',
        'letter-spacing:.5px',
        'background:#fff3e0',
        'color:#e65100',
        'width:fit-content',
      ].join(';');
      dom.modeIndicator.innerHTML =
        '📊 Historical Analysis — Actual vs Predicted will be shown';
    } else {
      dom.modeIndicator.style.cssText = [
        'display:block',
        'margin-top:6px',
        'padding:4px 12px',
        'border-radius:20px',
        'font-size:11px',
        'font-weight:700',
        'letter-spacing:.5px',
        'background:#e8f5e9',
        'color:#2e7d32',
        'width:fit-content',
      ].join(';');
      dom.modeIndicator.innerHTML = '🔮 Future Forecast Mode';
    }
  }

  function initDatePicker() {
    dom.forecastDate.addEventListener('change', () => {
      const val = dom.forecastDate.value;
      if (!val) return;

      const parts   = val.split('-');
      const dateObj = new Date(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]));

      state.date       = val;
      state.dateObj    = dateObj;
      state.month      = dateObj.getMonth() + 1;
      state.dayOfWeek  = dateObj.getDay();
      state.isHistorical = isHistoricalDate(dateObj);

      dom.chipMonthText.textContent = MONTHS[dateObj.getMonth()];
      dom.chipMonth.classList.add('visible');
      dom.chipDayText.textContent = DAYS[dateObj.getDay()];
      dom.chipDay.classList.add('visible');

      updateModeIndicator();
      tryFetchTemperature();
    });
  }

  // ============================================
  // Hour Slider
  // ============================================
  function initHourSlider() {
    function updateHourDisplay() {
      const h   = parseInt(dom.hourSlider.value);
      state.hour = h;
      const ampm = h >= 12 ? 'PM' : 'AM';
      const h12  = h % 12 || 12;
      dom.hourDisplayText.textContent = `${h12}:00 ${ampm}`;
      const isPeak = h >= 19 && h <= 22;
      dom.hourDisplay.classList.toggle('peak', isPeak);
    }
    dom.hourSlider.addEventListener('input', updateHourDisplay);
    updateHourDisplay();
  }

  // ============================================
  // Temperature Auto-Fetch
  // ============================================
  let tempFetchController = null;

  function tryFetchTemperature() {
    if (!state.region || !state.date) return;

    if (tempFetchController) tempFetchController.abort();
    tempFetchController = new AbortController();

    setTempStatus('loading', 'Fetching temperature…');

    const url = `${API_BASE}/get_temperature?region=${encodeURIComponent(state.region)}&date=${encodeURIComponent(state.date)}&hour=${state.hour}`;

    fetch(url, { signal: tempFetchController.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data.temperature_celsius != null) {
          dom.tempInput.value   = data.temperature_celsius;
          state.temperature     = data.temperature_celsius;
          state.tempAutoFetched = true;
          state.regionalHourlyTemperatures = data.regional_hourly_temperatures || [];
          state.nationalHourlyTemperatures = data.national_hourly_temperatures || [];
          const src = data.source && data.source.includes('average')
            ? `Monthly avg — ${data.city}`
            : `Auto-fetched — ${data.city}`;
          setTempStatus('auto', src);
        } else {
          throw new Error('No temperature data');
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        state.tempAutoFetched = false;
        state.regionalHourlyTemperatures = [];
        state.nationalHourlyTemperatures = [];
        setTempStatus('error', 'Weather API unavailable — please enter temperature manually');
      });
  }

  function setTempStatus(type, text) {
    dom.tempStatus.className = `temp-status ${type}`;
    const icons = {
      auto:    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/></svg>',
      manual:  '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>',
      error:   '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
      loading: '<div class="temp-spinner"></div>',
    };
    dom.tempStatus.innerHTML = `${icons[type] || ''}<span id="temp-status-text">${text}</span>`;
  }

  function initTempInput() {
    dom.tempInput.addEventListener('input', () => {
      state.temperature = parseFloat(dom.tempInput.value) || null;
      if (state.tempAutoFetched) {
        state.tempAutoFetched = false;
        setTempStatus('manual', 'Manually entered');
        state.regionalHourlyTemperatures = [];
        state.nationalHourlyTemperatures = [];
      }
    });
  }

  // ============================================
  // Holiday Toggle
  // ============================================
  function initHolidayToggle() {
    dom.holidayCheckbox.addEventListener('change', () => {
      state.isHoliday = dom.holidayCheckbox.checked;
      dom.holidayWarning.classList.toggle('visible', state.isHoliday);
    });
  }

  // ============================================
  // Forecast Submission
  // ============================================
  function initForecastButton() {
    dom.forecastBtn.addEventListener('click', () => {
      if (!state.region) { shakeElement(dom.regionPills); return; }
      if (!state.date)   { shakeElement(dom.forecastDate); return; }
      if (state.temperature == null || isNaN(state.temperature)) {
        shakeElement(dom.tempInput); return;
      }

      dom.forecastBtn.classList.add('loading');
      dom.forecastBtn.innerHTML = `
        <div class="temp-spinner" style="border-color:rgba(255,255,255,0.3);border-top-color:white;"></div>
        ${state.isHistorical ? 'Analysing Data…' : 'Processing Forecast…'}
      `;
      dom.progressWrapper.classList.add('active');
      dom.progressBar.classList.remove('filling');
      requestAnimationFrame(() => { dom.progressBar.classList.add('filling'); });

      setTimeout(() => { callForecastAPI(); }, 1500);
    });
  }

  function shakeElement(el) {
    el.style.animation = 'none';
    el.offsetHeight;
    el.style.animation = 'shake 0.4s ease';
    setTimeout(() => { el.style.animation = ''; }, 400);
  }

  const shakeStyle = document.createElement('style');
  shakeStyle.textContent = `
    @keyframes shake {
      0%,100%{transform:translateX(0)}
      20%{transform:translateX(-8px)}
      40%{transform:translateX(8px)}
      60%{transform:translateX(-4px)}
      80%{transform:translateX(4px)}
    }
    .mode-indicator { transition: all 0.3s ease; }
  `;
  document.head.appendChild(shakeStyle);

  async function callForecastAPI() {
    const body = {
      region:     state.region,
      date:       state.date,
      month:      state.month,
      day_of_week: state.dayOfWeek === 0 ? 6 : state.dayOfWeek - 1,
      hour:       state.hour,
      temperature: state.temperature,
      is_holiday: state.isHoliday,
      national_hourly_temperatures: state.nationalHourlyTemperatures,
    };

    try {
      // Always fetch prediction
      const predRes  = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });
      if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
      const predData = await predRes.json();
      state.lastPrediction = predData;

      let actualData = null;

      // If historical, also fetch actual values
      if (state.isHistorical) {
        try {
          const histRes = await fetch(
            `${API_BASE}/get_history?date=${encodeURIComponent(state.date)}&region=${encodeURIComponent(state.region)}`
          );
          if (histRes.ok) {
            const h = await histRes.json();
            if (h.available && h.hourly_actual && h.hourly_actual.length > 0) {
              actualData = h.hourly_actual;
            }
          }
        } catch (_) {
          // Actual data unavailable — forecast-only mode is fine
        }
      }

      renderResults(predData, actualData);

    } catch (err) {
      console.error('Prediction API error:', err);
      alert('Failed to get forecast. Please ensure the backend server is running at http://localhost:5000');
    } finally {
      resetForecastButton();
    }
  }

  function resetForecastButton() {
    dom.forecastBtn.classList.remove('loading');
    dom.forecastBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z"/></svg>
      Forecast Demand
    `;
    dom.progressWrapper.classList.remove('active');
    dom.progressBar.classList.remove('filling');
    dom.progressBar.style.width = '0%';
  }

  // ============================================
  // Results Rendering
  // ============================================
  function renderResults(data, actualHourly) {
    const {
      predicted_demand_gw: demand,
      confidence_low:      low,
      confidence_high:     high,
      hourly_forecast:     hourly,
      regional_comparison: regional,
    } = data;

    const dateParts = state.date.split('-');
    const dateObj   = new Date(parseInt(dateParts[0]), parseInt(dateParts[1])-1, parseInt(dateParts[2]));
    const dayName   = DAYS[dateObj.getDay()];
    const monthName = MONTHS[dateObj.getMonth()];
    const dateStr   = `${dayName}, ${dateObj.getDate()} ${monthName} ${dateObj.getFullYear()}`;

    const h12    = state.hour % 12 || 12;
    const ampm   = state.hour >= 12 ? 'PM' : 'AM';
    const timeStr = `${h12}:00 ${ampm}`;

    dom.predictionValue.textContent   = demand.toFixed(1);
    dom.predictionContext.textContent = `${REGION_LABELS[state.region] || state.region} | ${dateStr} | ${timeStr}`;
    dom.predictionRangeText.textContent = `Range: ${low.toFixed(1)} GW — ${high.toFixed(1)} GW`;

    // Status badge
    const avg       = SEASONAL_AVG[state.region] || 30;
    const pctAbove  = ((demand - avg) / avg) * 100;

    if (pctAbove > 20) {
      dom.statusBadge.className = 'status-badge critical';
      dom.statusBadge.querySelector('svg').innerHTML = '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>';
      dom.statusBadgeText.textContent = 'Critical Peak — Action Required';
    } else if (pctAbove > 10) {
      dom.statusBadge.className = 'status-badge high';
      dom.statusBadge.querySelector('svg').innerHTML = '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>';
      dom.statusBadgeText.textContent = 'High Load — Alert';
    } else {
      dom.statusBadge.className = 'status-badge normal';
      dom.statusBadge.querySelector('svg').innerHTML = '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>';
      dom.statusBadgeText.textContent = 'Normal Load';
    }

    // Charts
    renderHourlyChart(hourly, actualHourly, state.hour);
    renderRegionalChart(regional, state.region);
    renderInsights(hourly, actualHourly, demand, avg);

    // Show results
    dom.resultsSection.classList.remove('visible');
    void dom.resultsSection.offsetWidth;
    dom.resultsSection.classList.add('visible');
    setTimeout(() => {
      dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  // ============================================
  // Chart.js — 24-Hour Demand Curve
  // ============================================
  const verticalLinePlugin = {
    id: 'verticalLine',
    afterDatasetsDraw(chart) {
      const selectedHour = chart.options.plugins.verticalLine?.hour;
      if (selectedHour == null) return;
      const { ctx } = chart;
      const meta  = chart.getDatasetMeta(0);
      const point = meta.data[selectedHour];
      if (!point) return;
      const { x }   = point;
      const yAxis   = chart.scales.y;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#B7770D';
      ctx.lineWidth   = 2;
      ctx.moveTo(x, yAxis.top);
      ctx.lineTo(x, yAxis.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };
  Chart.register(verticalLinePlugin);

  function renderHourlyChart(predicted, actual, selectedHour) {
    if (hourlyChart) hourlyChart.destroy();

    const labels = Array.from({ length: 24 }, (_, i) => {
      const h    = i % 12 || 12;
      const ampm = i >= 12 ? 'PM' : 'AM';
      return `${h} ${ampm}`;
    });

    const hasActual = Array.isArray(actual) && actual.length === 24;
    const hasRegTemp = Array.isArray(state.regionalHourlyTemperatures) && state.regionalHourlyTemperatures.length === 24;
    const hasNatTemp = Array.isArray(state.nationalHourlyTemperatures) && state.nationalHourlyTemperatures.length === 24;

    // Update mode badge and legend
    if (hasActual) {
      dom.chartModeBadge.style.background = '#fff3e0';
      dom.chartModeBadge.style.color      = '#e65100';
      dom.chartModeBadge.textContent      = '📊 Actual vs Predicted';
      dom.legendActual.style.display      = 'flex';
    } else {
      dom.chartModeBadge.style.background = '#e8f5e9';
      dom.chartModeBadge.style.color      = '#2e7d32';
      dom.chartModeBadge.textContent      = '🔮 Forecast Only';
      dom.legendActual.style.display      = 'none';
    }

    if (hasRegTemp) {
      dom.legendRegionalTemp.style.display = 'flex';
    } else {
      dom.legendRegionalTemp.style.display = 'none';
    }

    if (hasNatTemp) {
      dom.legendNationalTemp.style.display = 'flex';
    } else {
      dom.legendNationalTemp.style.display = 'none';
    }

    // Highlight selected point on predicted line
    const pointBg = predicted.map((_, i) =>
      i === selectedHour ? '#B7770D' : 'transparent');
    const pointRadius = predicted.map((_, i) =>
      i === selectedHour ? 6 : 0);

    const datasets = [
      {
        label:              'Predicted (GW)',
        data:               predicted,
        fill:               !hasActual,
        borderColor:        '#1A3C6E',
        backgroundColor:    hasActual
          ? 'transparent'
          : createGradient(dom.hourlyChartCanvas, '#1A3C6E', 0.15),
        borderWidth:         2.5,
        tension:             0.4,
        pointBackgroundColor: pointBg,
        pointBorderColor:     pointBg,
        pointRadius:          pointRadius,
        pointHoverRadius:     6,
        borderDash:           hasActual ? [6, 3] : [],
        yAxisID:            'y',
      },
    ];

    if (hasActual) {
      datasets.push({
        label:              'Actual (GW)',
        data:               actual,
        fill:               true,
        borderColor:        '#e65100',
        backgroundColor:    createGradient(dom.hourlyChartCanvas, '#e65100', 0.08),
        borderWidth:         2,
        tension:             0.4,
        pointRadius:         0,
        pointHoverRadius:    5,
        borderDash:          [],
        yAxisID:            'y',
      });
    }

    if (hasRegTemp) {
      datasets.push({
        label:              'Regional Temp (°C)',
        data:               state.regionalHourlyTemperatures,
        fill:               false,
        borderColor:        '#e91e63',
        borderWidth:         1.5,
        tension:             0.4,
        pointRadius:         0,
        pointHoverRadius:    4,
        yAxisID:            'y1',
      });
    }

    if (hasNatTemp) {
      datasets.push({
        label:              'National Weighted Temp (°C)',
        data:               state.nationalHourlyTemperatures,
        fill:               false,
        borderColor:        '#00d4aa',
        borderWidth:         1.5,
        borderDash:          [4, 4],
        tension:             0.4,
        pointRadius:         0,
        pointHoverRadius:    4,
        yAxisID:            'y1',
      });
    }

    hourlyChart = new Chart(dom.hourlyChartCanvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           { duration: 1200, easing: 'easeOutQuart' },
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1A3C6E',
            titleFont:       { family: 'Inter', size: 12 },
            bodyFont:        { family: 'Inter', size: 13, weight: '600' },
            cornerRadius:    8,
            padding:         10,
            callbacks: {
              label: (ctx) => {
                const label = ctx.dataset.label;
                const val = ctx.parsed.y;
                if (label.includes('Temp')) {
                  return `${label}: ${val.toFixed(1)} °C`;
                }
                return `${label}: ${val.toFixed(2)} GW`;
              },
            },
          },
          verticalLine: { hour: selectedHour },
        },
        scales: {
          x: {
            grid:  { display: false },
            ticks: {
              font:        { family: 'Inter', size: 10 },
              color:       '#718096',
              maxRotation: 0,
              autoSkip:    true,
              maxTicksLimit: 12,
            },
          },
          y: {
            position: 'left',
            grid:   { color: '#F0F0F0', lineWidth: 1 },
            border: { display: false },
            ticks:  {
              font:     { family: 'Inter', size: 10 },
              color:    '#718096',
              callback: (v) => `${v.toFixed(1)} GW`,
            },
            title: {
              display: true,
              text: 'Demand (GW)',
              font: { family: 'Inter', size: 10, weight: '600' },
              color: '#718096',
            },
          },
          y1: {
            position: 'right',
            grid:   { display: false },
            border: { display: false },
            ticks:  {
              font:     { family: 'Inter', size: 10 },
              color:    '#718096',
              callback: (v) => `${v.toFixed(1)} °C`,
            },
            title: {
              display: true,
              text: 'Temperature (°C)',
              font: { family: 'Inter', size: 10, weight: '600' },
              color: '#718096',
            },
          },
        },
      },
    });
  }

  function createGradient(canvas, color, alpha) {
    const ctx      = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight || 280);
    gradient.addColorStop(0, hexToRgba(color, alpha));
    gradient.addColorStop(1, hexToRgba(color, 0.01));
    return gradient;
  }

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ============================================
  // Chart.js — Regional Comparison Bar Chart
  // ============================================
  function renderRegionalChart(regional, selectedRegion) {
    if (regionalChart) regionalChart.destroy();

    const regionKeys         = ['North','South','East','West','NorthEast'];
    const regionDisplayNames = ['North','South','East','West','North-East'];
    const values      = regionKeys.map((k) => regional[k] || 0);
    const colors      = regionKeys.map((k) => k === selectedRegion ? '#1A3C6E' : '#A8C8E8');
    const borderColors = regionKeys.map((k) => k === selectedRegion ? '#142E54' : '#8BB8DE');

    regionalChart = new Chart(dom.regionalChartCanvas, {
      type: 'bar',
      data: {
        labels:   regionDisplayNames,
        datasets: [{
          label:           'Demand (GW)',
          data:            values,
          backgroundColor: colors,
          borderColor:     borderColors,
          borderWidth:     1,
          borderRadius:    6,
          barThickness:    28,
        }],
      },
      options: {
        indexAxis:           'y',
        responsive:          true,
        maintainAspectRatio: false,
        animation:           { duration: 1000, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1A3C6E',
            titleFont:       { family: 'Inter', size: 12 },
            bodyFont:        { family: 'Inter', size: 13, weight: '600' },
            cornerRadius:    8,
            padding:         10,
            callbacks: {
              label: (ctx) => `${ctx.parsed.x.toFixed(2)} GW`,
            },
          },
        },
        scales: {
          x: {
            grid:   { color: '#F0F0F0', lineWidth: 1 },
            border: { display: false },
            ticks:  {
              font:     { family: 'Inter', size: 10 },
              color:    '#718096',
              callback: (v) => `${v.toFixed(1)} GW`,
            },
          },
          y: {
            grid:  { display: false },
            ticks: { font: { family: 'Inter', size: 12, weight: '600' }, color: '#1A1A1A' },
          },
        },
      },
    });
  }

  // ============================================
  // Insight Strip
  // ============================================
  function renderInsights(predicted, actual, demand, seasonalAvg) {
    // Peak hour (from predicted)
    let peakVal = 0, peakHour = 0;
    predicted.forEach((v, i) => { if (v > peakVal) { peakVal = v; peakHour = i; } });
    const peakH12  = peakHour % 12 || 12;
    const peakAmPm = peakHour >= 12 ? 'PM' : 'AM';
    dom.insightPeak.textContent = `${peakH12}:00 ${peakAmPm}`;

    // Temperature impact
    const baselineTemp = 25;
    const currentTemp  = state.temperature || 30;
    const tempDiffPct  = ((currentTemp - baselineTemp) / baselineTemp * 100 * 0.8).toFixed(0);
    const sign         = tempDiffPct >= 0 ? '+' : '';
    dom.insightTemp.textContent = `${sign}${tempDiffPct}% vs baseline`;

    // MAPE or YoY
    if (Array.isArray(actual) && actual.length === 24) {
      // Historical: show average error
      const errors = predicted.map((p, i) =>
        actual[i] !== 0 ? Math.abs((p - actual[i]) / actual[i]) * 100 : 0
      );
      const avgErr = errors.reduce((a, b) => a + b, 0) / errors.length;
      dom.insightYoy.textContent = `${avgErr.toFixed(1)}% avg error`;
      dom.insightYoy.closest('.insight-chip').querySelector('.insight-label').textContent = 'Model MAPE (24h)';
    } else {
      const yoyDelta = (demand * 0.025 + ((state.month || 1) - 6) * 0.3).toFixed(1);
      const yoySign  = yoyDelta >= 0 ? '+' : '';
      dom.insightYoy.textContent = `${yoySign}${yoyDelta} GW`;
      dom.insightYoy.closest('.insight-chip').querySelector('.insight-label').textContent = 'vs Last Year Same Day';
    }
  }

  // ============================================
  // Initialization
  // ============================================
  function init() {
    updateClock();
    setInterval(updateClock, 1000);

    initRegionPills();
    initDatePicker();
    initHourSlider();
    initTempInput();
    initHolidayToggle();
    initForecastButton();

    setTempStatus('manual', 'Enter manually');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
