/**
 * GridSense India — Application Logic
 *
 * Changes from previous version:
 *  1. Reads `all_regional_forecasts` (all 5 regions' 24h curves) from /predict.
 *  2. Reads `national_hourly_gw` (national 24h demand) from /predict.
 *  3. Renders a new "All Regions Comparison" line chart using those curves.
 *  4. Sinusoidal fallback completely removed — temperature state is only
 *     populated from the API or manual entry.
 *  5. Mode indicator logic and historical/future routing unchanged.
 */

(function () {
  'use strict';

  // ── Constants ────────────────────────────────────────────────────────────────
  const API_BASE   = 'http://localhost:5000';
  const DATASET_END = new Date('2024-04-30');
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
  const SEASONAL_AVG = {
    North: 52, South: 42, East: 28, West: 50, NorthEast: 8,
  };

  // Colour palette for the 5-region overlay chart
  const REGION_COLORS = {
    North:     '#1A3C6E',
    South:     '#e65100',
    East:      '#2e7d32',
    West:      '#6a1b9a',
    NorthEast: '#00838f',
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
    regionalHourlyTemperatures: [],   // selected region 24h temp curve
    nationalHourlyTemperatures: [],   // weighted national 24h temp curve
  };

  let hourlyChart      = null;
  let regionalChart    = null;
  let allRegionsChart  = null;   // new: all-5-regions demand overlay

  // ── DOM references ───────────────────────────────────────────────────────────
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
    allRegionsCanvas:    $('all-regions-chart'),   // new canvas (add to index.html)
    insightPeak:         $('insight-peak'),
    insightTemp:         $('insight-temp'),
    insightYoy:          $('insight-yoy'),
    chartModeBadge:      $('chart-mode-badge'),
    legendActual:        $('legend-actual'),
    legendRegionalTemp:  $('legend-regional-temp'),
    legendNationalTemp:  $('legend-national-temp'),
  };

  // ── Live IST clock ────────────────────────────────────────────────────────────
  function updateClock() {
    const now = new Date();
    const utc = now.getTime() + now.getTimezoneOffset() * 60000;
    const ist = new Date(utc + 5.5 * 3600000);
    const h   = ist.getHours(), m = ist.getMinutes(), s = ist.getSeconds();
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12  = h % 12 || 12;
    dom.headerTime.textContent =
      `${h12}:${pad(m)}:${pad(s)} ${ampm}`;
    dom.headerDate.textContent =
      `${DAYS[ist.getDay()]}, ${ist.getDate()} ${MONTHS[ist.getMonth()]} ${ist.getFullYear()}`;
  }
  const pad = (n) => String(n).padStart(2, '0');

  // ── Region pills ──────────────────────────────────────────────────────────────
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

  // ── Date picker ───────────────────────────────────────────────────────────────
  function isHistoricalDate(dateObj) {
    const d   = new Date(dateObj.getFullYear(), dateObj.getMonth(), dateObj.getDate());
    const end = new Date(DATASET_END.getFullYear(), DATASET_END.getMonth(), DATASET_END.getDate());
    return d <= end;
  }

  function updateModeIndicator() {
    if (!state.date) { dom.modeIndicator.style.display = 'none'; return; }
    const base = 'display:block;margin-top:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px;width:fit-content;';
    if (state.isHistorical) {
      dom.modeIndicator.style.cssText = base + 'background:#fff3e0;color:#e65100;';
      dom.modeIndicator.innerHTML = '📊 Historical Analysis — Actual vs Predicted will be shown';
    } else {
      dom.modeIndicator.style.cssText = base + 'background:#e8f5e9;color:#2e7d32;';
      dom.modeIndicator.innerHTML = '🔮 Future Forecast Mode';
    }
  }

  function initDatePicker() {
    dom.forecastDate.addEventListener('change', () => {
      const val = dom.forecastDate.value;
      if (!val) return;
      const [y, mo, d] = val.split('-').map(Number);
      const dateObj = new Date(y, mo - 1, d);
      state.date        = val;
      state.dateObj     = dateObj;
      state.month       = mo;
      state.dayOfWeek   = dateObj.getDay();
      state.isHistorical = isHistoricalDate(dateObj);
      dom.chipMonthText.textContent = MONTHS[mo - 1];
      dom.chipMonth.classList.add('visible');
      dom.chipDayText.textContent = DAYS[dateObj.getDay()];
      dom.chipDay.classList.add('visible');
      updateModeIndicator();
      tryFetchTemperature();
    });
  }

  // ── Hour slider ───────────────────────────────────────────────────────────────
  function initHourSlider() {
    function update() {
      const h = parseInt(dom.hourSlider.value);
      state.hour = h;
      const ampm = h >= 12 ? 'PM' : 'AM';
      dom.hourDisplayText.textContent = `${h % 12 || 12}:00 ${ampm}`;
      dom.hourDisplay.classList.toggle('peak', h >= 19 && h <= 22);
    }
    dom.hourSlider.addEventListener('input', update);
    update();
  }

  // ── Temperature auto-fetch ────────────────────────────────────────────────────
  let tempFetchController = null;

  function tryFetchTemperature() {
    if (!state.region || !state.date) return;
    if (tempFetchController) tempFetchController.abort();
    tempFetchController = new AbortController();
    setTempStatus('loading', 'Fetching temperature…');

    const url = `${API_BASE}/get_temperature?region=${encodeURIComponent(state.region)}&date=${encodeURIComponent(state.date)}&hour=${state.hour}`;

    fetch(url, { signal: tempFetchController.signal })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((data) => {
        if (data.temperature_celsius != null) {
          dom.tempInput.value   = data.temperature_celsius;
          state.temperature     = data.temperature_celsius;
          state.tempAutoFetched = true;
          // Store both curves — these are passed to /predict and rendered on chart
          state.regionalHourlyTemperatures = data.regional_hourly_temperatures || [];
          state.nationalHourlyTemperatures = data.national_hourly_temperatures  || [];
          const src = (data.source || '').includes('average')
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
        setTempStatus('error', 'Weather API unavailable — enter temperature manually');
      });
  }

  function setTempStatus(type, text) {
    const icons = {
      auto:    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/></svg>',
      manual:  '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>',
      error:   '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
      loading: '<div class="temp-spinner"></div>',
    };
    dom.tempStatus.className = `temp-status ${type}`;
    dom.tempStatus.innerHTML = `${icons[type] || ''}<span id="temp-status-text">${text}</span>`;
  }

  function initTempInput() {
    dom.tempInput.addEventListener('input', () => {
      state.temperature = parseFloat(dom.tempInput.value) || null;
      if (state.tempAutoFetched) {
        state.tempAutoFetched = false;
        setTempStatus('manual', 'Manually entered');
        // Clear stored curves — manual entry has no 24h curve
        state.regionalHourlyTemperatures = [];
        state.nationalHourlyTemperatures = [];
      }
    });
  }

  // ── Holiday toggle ────────────────────────────────────────────────────────────
  function initHolidayToggle() {
    dom.holidayCheckbox.addEventListener('change', () => {
      state.isHoliday = dom.holidayCheckbox.checked;
      dom.holidayWarning.classList.toggle('visible', state.isHoliday);
    });
  }

  // ── Forecast button ───────────────────────────────────────────────────────────
  function initForecastButton() {
    dom.forecastBtn.addEventListener('click', () => {
      if (!state.region)   { shakeElement(dom.regionPills); return; }
      if (!state.date)     { shakeElement(dom.forecastDate); return; }
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
      requestAnimationFrame(() => dom.progressBar.classList.add('filling'));
      setTimeout(callForecastAPI, 1500);
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
    }`;
  document.head.appendChild(shakeStyle);

  // ── API call ──────────────────────────────────────────────────────────────────
  async function callForecastAPI() {
    const body = {
      region:      state.region,
      date:        state.date,
      month:       state.month,
      day_of_week: state.dayOfWeek === 0 ? 6 : state.dayOfWeek - 1,
      hour:        state.hour,
      temperature: state.temperature,
      is_holiday:  state.isHoliday,
      // Send the 24-value national weighted curve if available (future dates via API)
      national_hourly_temperatures: state.nationalHourlyTemperatures,
    };

    try {
      const predRes = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });
      if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
      const predData = await predRes.json();
      state.lastPrediction = predData;

      let actualData = null;

      if (state.isHistorical) {
        try {
          const histRes = await fetch(
            `${API_BASE}/get_history?date=${encodeURIComponent(state.date)}&region=${encodeURIComponent(state.region)}`
          );
          if (histRes.ok) {
            const h = await histRes.json();
            if (h.available && h.hourly_actual?.length > 0) actualData = h.hourly_actual;
          }
        } catch (_) { /* actual unavailable — forecast-only is fine */ }
      }

      renderResults(predData, actualData);

    } catch (err) {
      console.error('Prediction API error:', err);
      alert('Failed to get forecast. Ensure the backend is running at http://localhost:5000');
    } finally {
      resetForecastButton();
    }
  }

  function resetForecastButton() {
    dom.forecastBtn.classList.remove('loading');
    dom.forecastBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L3 14h9l-1 10 10-12h-9l1-10z"/></svg>
      Forecast Demand`;
    dom.progressWrapper.classList.remove('active');
    dom.progressBar.classList.remove('filling');
    dom.progressBar.style.width = '0%';
  }

  // ── Results rendering ─────────────────────────────────────────────────────────
  function renderResults(data, actualHourly) {
    const {
      predicted_demand_gw:  demand,
      confidence_low:       low,
      confidence_high:      high,
      hourly_forecast:      hourly,
      regional_comparison:  regional,
      all_regional_forecasts: allRegional,  // new: all 5 regions 24h
      national_hourly_gw:   nationalGw,     // new: national 24h
    } = data;

    const [y, mo, d] = state.date.split('-').map(Number);
    const dateObj = new Date(y, mo - 1, d);
    const dateStr = `${DAYS[dateObj.getDay()]}, ${d} ${MONTHS[mo - 1]} ${y}`;
    const h12 = state.hour % 12 || 12;
    const ampm = state.hour >= 12 ? 'PM' : 'AM';

    dom.predictionValue.textContent     = demand.toFixed(1);
    dom.predictionContext.textContent   = `${REGION_LABELS[state.region] || state.region} | ${dateStr} | ${h12}:00 ${ampm}`;
    dom.predictionRangeText.textContent = `Range: ${low.toFixed(1)} GW — ${high.toFixed(1)} GW`;

    // Status badge
    const avg      = SEASONAL_AVG[state.region] || 30;
    const pctAbove = ((demand - avg) / avg) * 100;
    if (pctAbove > 20) {
      dom.statusBadge.className = 'status-badge critical';
      dom.statusBadgeText.textContent = 'Critical Peak — Action Required';
    } else if (pctAbove > 10) {
      dom.statusBadge.className = 'status-badge high';
      dom.statusBadgeText.textContent = 'High Load — Alert';
    } else {
      dom.statusBadge.className = 'status-badge normal';
      dom.statusBadgeText.textContent = 'Normal Load';
    }

    renderHourlyChart(hourly, actualHourly, state.hour);
    renderRegionalChart(regional, state.region);

    // Render all-regions overlay only when backend returned the full curves
    if (allRegional && Object.keys(allRegional).length === 5) {
      renderAllRegionsChart(allRegional, state.region, state.hour);
    }

    renderInsights(hourly, actualHourly, demand, avg);

    dom.resultsSection.classList.remove('visible');
    void dom.resultsSection.offsetWidth;
    dom.resultsSection.classList.add('visible');
    setTimeout(() => dom.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
  }

  // ── 24-hour demand + temperature chart ───────────────────────────────────────
  const verticalLinePlugin = {
    id: 'verticalLine',
    afterDatasetsDraw(chart) {
      const h = chart.options.plugins.verticalLine?.hour;
      if (h == null) return;
      const { ctx } = chart;
      const pt = chart.getDatasetMeta(0).data[h];
      if (!pt) return;
      const yAxis = chart.scales.y;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#B7770D';
      ctx.lineWidth   = 2;
      ctx.moveTo(pt.x, yAxis.top);
      ctx.lineTo(pt.x, yAxis.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };
  Chart.register(verticalLinePlugin);

  function renderHourlyChart(predicted, actual, selectedHour) {
    if (hourlyChart) hourlyChart.destroy();

    const labels    = Array.from({ length: 24 }, (_, i) => `${i % 12 || 12} ${i >= 12 ? 'PM' : 'AM'}`);
    const hasActual  = Array.isArray(actual) && actual.length === 24;
    const hasRegTemp = state.regionalHourlyTemperatures.length === 24;
    const hasNatTemp = state.nationalHourlyTemperatures.length === 24;

    // Update badge and legend visibility
    if (hasActual) {
      dom.chartModeBadge.style.cssText = 'background:#fff3e0;color:#e65100;';
      dom.chartModeBadge.textContent   = '📊 Actual vs Predicted';
      dom.legendActual.style.display   = 'flex';
    } else {
      dom.chartModeBadge.style.cssText = 'background:#e8f5e9;color:#2e7d32;';
      dom.chartModeBadge.textContent   = '🔮 Forecast Only';
      dom.legendActual.style.display   = 'none';
    }
    if (dom.legendRegionalTemp) dom.legendRegionalTemp.style.display = hasRegTemp ? 'flex' : 'none';
    if (dom.legendNationalTemp) dom.legendNationalTemp.style.display = hasNatTemp ? 'flex' : 'none';

    const pointBg     = predicted.map((_, i) => i === selectedHour ? '#B7770D' : 'transparent');
    const pointRadius = predicted.map((_, i) => i === selectedHour ? 6 : 0);

    const datasets = [{
      label:               'Predicted (GW)',
      data:                predicted,
      fill:                !hasActual,
      borderColor:         '#1A3C6E',
      backgroundColor:     hasActual ? 'transparent' : createGradient(dom.hourlyChartCanvas, '#1A3C6E', 0.15),
      borderWidth:         2.5,
      tension:             0.4,
      pointBackgroundColor: pointBg,
      pointBorderColor:     pointBg,
      pointRadius,
      pointHoverRadius:    6,
      borderDash:          hasActual ? [6, 3] : [],
      yAxisID:             'y',
    }];

    if (hasActual) {
      datasets.push({
        label:           'Actual (GW)',
        data:            actual,
        fill:            true,
        borderColor:     '#e65100',
        backgroundColor: createGradient(dom.hourlyChartCanvas, '#e65100', 0.08),
        borderWidth:     2,
        tension:         0.4,
        pointRadius:     0,
        pointHoverRadius: 5,
        yAxisID:         'y',
      });
    }

    if (hasRegTemp) {
      datasets.push({
        label:       'Regional Temp (°C)',
        data:        state.regionalHourlyTemperatures,
        fill:        false,
        borderColor: '#e91e63',
        borderWidth: 1.5,
        tension:     0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
        yAxisID:     'y1',
      });
    }

    if (hasNatTemp) {
      datasets.push({
        label:       'National Weighted Temp (°C)',
        data:        state.nationalHourlyTemperatures,
        fill:        false,
        borderColor: '#00d4aa',
        borderWidth: 1.5,
        borderDash:  [4, 4],
        tension:     0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
        yAxisID:     'y1',
      });
    }

    hourlyChart = new Chart(dom.hourlyChartCanvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation:   { duration: 1200, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1A3C6E', cornerRadius: 8, padding: 10,
            callbacks: {
              label: (ctx) => {
                const v = ctx.parsed.y;
                return ctx.dataset.label.includes('Temp')
                  ? `${ctx.dataset.label}: ${v.toFixed(1)} °C`
                  : `${ctx.dataset.label}: ${v.toFixed(2)} GW`;
              },
            },
          },
          verticalLine: { hour: selectedHour },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12, color: '#718096', font: { size: 10 } } },
          y: {
            position: 'left',
            grid:   { color: '#F0F0F0' },
            border: { display: false },
            ticks:  { callback: (v) => `${v.toFixed(1)} GW`, color: '#718096', font: { size: 10 } },
            title:  { display: true, text: 'Demand (GW)', color: '#718096', font: { size: 10, weight: '600' } },
          },
          y1: {
            position: 'right',
            grid:   { display: false },
            border: { display: false },
            ticks:  { callback: (v) => `${v.toFixed(1)} °C`, color: '#718096', font: { size: 10 } },
            title:  { display: true, text: 'Temperature (°C)', color: '#718096', font: { size: 10, weight: '600' } },
          },
        },
      },
    });
  }

  // ── All-5-regions 24h demand overlay ─────────────────────────────────────────
  function renderAllRegionsChart(allRegional, selectedRegion, selectedHour) {
    if (!dom.allRegionsCanvas) return;
    if (allRegionsChart) allRegionsChart.destroy();

    const labels   = Array.from({ length: 24 }, (_, i) => `${i % 12 || 12} ${i >= 12 ? 'PM' : 'AM'}`);
    const regions  = ['North', 'West', 'South', 'East', 'NorthEast'];
    const datasets = regions.map((r) => ({
      label:       REGION_LABELS[r] || r,
      data:        allRegional[r] || [],
      fill:        false,
      borderColor: REGION_COLORS[r],
      borderWidth: r === selectedRegion ? 3 : 1.5,
      borderDash:  r === selectedRegion ? [] : [4, 3],
      tension:     0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
    }));

    // Vertical line at selected hour
    const vertLinePlugin = {
      id: 'allRegionsVLine',
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const pt = chart.getDatasetMeta(0).data[selectedHour];
        if (!pt) return;
        const yAxis = chart.scales.y;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#B7770D';
        ctx.lineWidth   = 2;
        ctx.moveTo(pt.x, yAxis.top);
        ctx.lineTo(pt.x, yAxis.bottom);
        ctx.stroke();
        ctx.restore();
      },
    };

    allRegionsChart = new Chart(dom.allRegionsCanvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation:   { duration: 1000, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display:  true,
            position: 'top',
            labels:   { usePointStyle: true, pointStyleWidth: 12, color: '#1A1A1A', font: { size: 11 } },
          },
          tooltip: {
            backgroundColor: '#1A3C6E', cornerRadius: 8, padding: 10,
            callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} GW` },
          },
          allRegionsVLine: {},
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12, color: '#718096', font: { size: 10 } } },
          y: {
            grid:   { color: '#F0F0F0' },
            border: { display: false },
            ticks:  { callback: (v) => `${v.toFixed(1)} GW`, color: '#718096', font: { size: 10 } },
            title:  { display: true, text: 'Demand (GW)', color: '#718096', font: { size: 10, weight: '600' } },
          },
        },
      },
      plugins: [vertLinePlugin],
    });
  }

  // ── Regional comparison bar chart ─────────────────────────────────────────────
  function renderRegionalChart(regional, selectedRegion) {
    if (regionalChart) regionalChart.destroy();

    const regionKeys  = ['North', 'South', 'East', 'West', 'NorthEast'];
    const regionNames = ['North', 'South', 'East', 'West', 'North-East'];
    const values      = regionKeys.map((k) => regional[k] || 0);
    const colors      = regionKeys.map((k) => k === selectedRegion ? '#1A3C6E' : '#A8C8E8');

    regionalChart = new Chart(dom.regionalChartCanvas, {
      type: 'bar',
      data: {
        labels: regionNames,
        datasets: [{
          label:           'Demand (GW)',
          data:            values,
          backgroundColor: colors,
          borderColor:     regionKeys.map((k) => k === selectedRegion ? '#142E54' : '#8BB8DE'),
          borderWidth:     1,
          borderRadius:    6,
          barThickness:    28,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        animation:  { duration: 1000, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1A3C6E', cornerRadius: 8, padding: 10,
            callbacks: { label: (ctx) => `${ctx.parsed.x.toFixed(2)} GW` },
          },
        },
        scales: {
          x: { grid: { color: '#F0F0F0' }, border: { display: false }, ticks: { callback: (v) => `${v.toFixed(1)} GW`, color: '#718096', font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: '#1A1A1A', font: { size: 12, weight: '600' } } },
        },
      },
    });
  }

  // ── Insight strip ─────────────────────────────────────────────────────────────
  function renderInsights(predicted, actual, demand, seasonalAvg) {
    let peakVal = 0, peakHour = 0;
    predicted.forEach((v, i) => { if (v > peakVal) { peakVal = v; peakHour = i; } });
    dom.insightPeak.textContent = `${peakHour % 12 || 12}:00 ${peakHour >= 12 ? 'PM' : 'AM'}`;

    const baselineTemp = 25;
    const currentTemp  = state.temperature || 30;
    const diffPct      = ((currentTemp - baselineTemp) / baselineTemp * 100 * 0.8).toFixed(0);
    dom.insightTemp.textContent = `${diffPct >= 0 ? '+' : ''}${diffPct}% vs baseline`;

    if (Array.isArray(actual) && actual.length === 24) {
      const errors = predicted.map((p, i) =>
        actual[i] !== 0 ? Math.abs((p - actual[i]) / actual[i]) * 100 : 0
      );
      const avgErr = errors.reduce((a, b) => a + b, 0) / errors.length;
      dom.insightYoy.textContent = `${avgErr.toFixed(1)}% avg error`;
      const label = dom.insightYoy.closest('.insight-chip')?.querySelector('.insight-label');
      if (label) label.textContent = 'Model MAPE (24h)';
    } else {
      const delta = (demand * 0.025 + ((state.month || 1) - 6) * 0.3).toFixed(1);
      dom.insightYoy.textContent = `${delta >= 0 ? '+' : ''}${delta} GW`;
    }
  }

  // ── Gradient helper ───────────────────────────────────────────────────────────
  function createGradient(canvas, color, alpha) {
    const ctx      = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight || 280);
    gradient.addColorStop(0, hexToRgba(color, alpha));
    gradient.addColorStop(1, hexToRgba(color, 0.01));
    return gradient;
  }

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ── Init ──────────────────────────────────────────────────────────────────────
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

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', init)
    : init();
})();