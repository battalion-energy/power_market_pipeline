# Per-Node Price Forecasting Architecture

**Location Matters - West Texas ≠ Houston**

ERCOT has ~4,000 settlement points with vastly different price dynamics. A battery in West Texas (wind-heavy) faces completely different prices than Houston (load center, gas-heavy).

**Critical Insight:** We need separate models for each major node/zone, accounting for local generation, congestion, and transmission constraints.

---

## 🎯 Problem Statement

### Why Per-Node Models?

**Settlement Point Price (SPP) = System Lambda + Congestion Component + Loss Component**

```
SPP[node] = System_Lambda + Congestion[node] + Loss[node]
```

**System Lambda:** Common across all nodes (system marginal cost)
**Congestion:** Node-specific (transmission constraints)
**Loss:** Node-specific (electrical losses from generation to load)

### Example Price Differences (Same Hour)

| Settlement Point | Zone | DA Price | RT Price | Why Different? |
|------------------|------|----------|----------|----------------|
| **HB_HOUSTON** | Houston | $45/MWh | $48/MWh | Load center, well-connected |
| **HB_WEST** | West Texas | $35/MWh | $55/MWh | Wind curtailment, then spike |
| **HB_SOUTH** | South Texas | $50/MWh | $52/MWh | Constrained import, gas generation |
| **HB_NORTH** | North Texas | $42/MWh | $45/MWh | DFW load + wind |

**Same system, same hour, $15/MWh spread!**

---

## 📍 ERCOT Settlement Point Zones

### Major Load Zones (Hub Nodes)

1. **HB_HOUSTON** (Houston Hub)
   - Largest load center (5M+ people)
   - Gas-heavy generation
   - Well-connected transmission
   - Price characteristics: Stable, follows system lambda closely
   - Congestion: Low (usually)

2. **HB_NORTH** (North Hub)
   - DFW metroplex
   - Mix of wind + gas
   - Good transmission
   - Price characteristics: Moderate volatility
   - Congestion: Moderate during peak

3. **HB_SOUTH** (South Hub)
   - South Texas
   - Gas + some solar
   - Transmission constraints to rest of grid
   - Price characteristics: Higher volatility
   - Congestion: High during summer

4. **HB_WEST** (West Hub)
   - West Texas
   - HEAVY wind generation (15+ GW capacity)
   - Limited transmission to load centers
   - Price characteristics: Negative prices (curtailment), then spikes
   - Congestion: Extreme (both directions)

5. **HB_PAN** (Panhandle)
   - Far north Texas
   - Wind + some gas
   - Remote from load centers
   - Price characteristics: Follows West Texas patterns
   - Congestion: High export constraints

### Individual Battery Nodes

- **MOSS1** (Moss Landing) → HB_SOUTH zone
- **GIBBONSCR** (Gibbons Creek) → HB_NORTH zone
- **ANGLETON** → HB_HOUSTON zone

---

## 🧠 Model Architecture

### Hierarchical Forecasting Approach

```
Level 1: System Lambda Forecast (System-Wide)
         ↓
Level 2a: Zone Congestion Models (5 zones)
         ↓
Level 2b: Loss Models (5 zones)
         ↓
Level 3: Node-Specific Adjustments (per battery location)
         ↓
Final: SPP[node] = Lambda + Congestion[zone] + Loss[zone] + Local[node]
```

### Model 1: System Lambda Forecasting

**Purpose:** Predict system-wide marginal cost

**Architecture:** LSTM-Attention (existing Model 1)

**Features:**
- System-wide net load
- Total wind generation
- Total solar generation
- Fuel prices (gas)
- Hour of day, season

**Output:** 24-hour DA system lambda forecast

**Why First:** System lambda is common component for all nodes

---

### Model 2: Per-Zone Congestion Forecasting

**Purpose:** Predict congestion at each zone

**Architecture:** Separate model for each zone (5 models)

#### Model 2a: Houston Congestion
```python
class HoustonCongestionModel(nn.Module):
    """
    Houston zone congestion model.

    Houston characteristics:
    - Large load center
    - Usually uncongested (well-connected)
    - Congestion only during extreme heat
    """

    def __init__(self):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 24)  # 24 hourly congestion values
        )

    def forward(self, features):
        """
        Features specific to Houston congestion:
        - Houston load
        - Import from West Texas (wind)
        - Import from South Texas
        - Local gas generation capacity
        - Temperature (Houston)
        """
        # ... LSTM + Attention processing ...
        return congestion_forecast  # (24,) hourly values
```

#### Model 2b: West Texas Congestion
```python
class WestTexasCongestionModel(nn.Module):
    """
    West Texas zone congestion model.

    West Texas characteristics:
    - MASSIVE wind generation (15+ GW)
    - Limited transmission export
    - Frequent congestion (both directions):
      * Negative congestion (too much wind, can't export)
      * Positive congestion (wind dies, load centers need imports)
    """

    def __init__(self):
        # More complex than Houston - needs to handle bidirectional congestion
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        self.congestion_direction_classifier = nn.Linear(256, 3)  # Export / None / Import
        self.congestion_magnitude = nn.Linear(256, 24)

    def forward(self, features):
        """
        Features specific to West Texas congestion:
        - West Texas wind generation
        - Wind forecast error (CRITICAL)
        - Export flow to Houston/North
        - Transmission line ratings
        - Hour of day (wind patterns)
        - Season (summer = export constrained, winter = import)
        """

        # Classify direction
        direction = self.congestion_direction_classifier(encoded)
        # Predict magnitude
        magnitude = self.congestion_magnitude(encoded)

        # Combine
        congestion = direction * magnitude

        return congestion_forecast
```

**Key Insight:** West Texas model needs to predict **negative prices** during high wind + low load (curtailment).

---

### Model 3: Per-Zone Loss Forecasting

**Purpose:** Predict electrical losses at each zone

**Architecture:** Simpler than congestion (losses more predictable)

```python
class ZoneLossModel(nn.Module):
    """
    Electrical loss model for a zone.

    Losses depend on:
    - Distance from generation
    - Load level (losses increase with flow²)
    - Voltage levels
    """

    def __init__(self, zone_name):
        self.zone = zone_name
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 24)  # 24 hourly loss values
        )

    def forward(self, features):
        """
        Features for loss prediction:
        - Total system load
        - Zone load
        - Net import/export
        - Generation mix (affects power flow patterns)
        """
        return loss_forecast
```

---

### Model 4: Node-Specific Adjustments

**Purpose:** Fine-tune forecast for specific battery node

**Why?** Even within a zone, nodes can differ due to local constraints.

```python
class NodeSpecificAdjustment(nn.Module):
    """
    Learn node-specific price adjustments.

    Example: MOSS1 vs. another battery in same zone
    might have slightly different prices due to local feeder constraints.
    """

    def __init__(self, node_name):
        self.node = node_name
        # Small network - adjustments are usually small
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24)
        )

    def forward(self, features):
        """
        Learn node-specific patterns from historical data.

        This captures:
        - Local feeder constraints
        - Proximity to specific generators
        - Historical basis differentials
        """
        return node_adjustment
```

---

## 🎯 Complete Per-Node Price Forecast

### Forward Pass

```python
def forecast_node_price(node_name: str, features: dict) -> np.ndarray:
    """
    Generate 24-hour DA price forecast for specific node.

    Args:
        node_name: e.g., "MOSS1_UNIT1", "HB_HOUSTON"
        features: Market conditions, forecasts, weather

    Returns:
        24-hour price forecast at that specific node
    """

    # Step 1: Forecast system lambda (common to all)
    system_lambda = system_lambda_model(features)  # (24,)

    # Step 2: Determine zone
    zone = get_zone_for_node(node_name)  # "HB_SOUTH", "HB_HOUSTON", etc.

    # Step 3: Forecast zone congestion
    if zone == "HB_HOUSTON":
        congestion = houston_congestion_model(features)
    elif zone == "HB_WEST":
        congestion = west_texas_congestion_model(features)
    elif zone == "HB_SOUTH":
        congestion = south_texas_congestion_model(features)
    elif zone == "HB_NORTH":
        congestion = north_texas_congestion_model(features)
    elif zone == "HB_PAN":
        congestion = panhandle_congestion_model(features)

    # Step 4: Forecast zone losses
    losses = zone_loss_models[zone](features)

    # Step 5: Node-specific adjustment
    node_adjustment = node_specific_models[node_name](features)

    # Step 6: Combine
    node_price = system_lambda + congestion + losses + node_adjustment

    return node_price
```

---

## 📊 Training Data Requirements

### Per-Zone Data

Each zone model needs:

**Training Data:**
- Historical DA/RT prices at zone hub (2019-2024)
- Zone-specific load
- Zone-specific generation
- Import/export flows
- Weather for zone
- Congestion events (when did congestion occur?)

**Data Files:**
```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/
├── DA_Hourly_LMPs_by_zone_2019.parquet
├── DA_Hourly_LMPs_by_zone_2020.parquet
├── ...
├── RT_5min_LMPs_by_zone_2019.parquet
├── ...
```

**Zone Mapping:**
```python
ZONE_MAPPING = {
    # Houston Hub
    "HB_HOUSTON": ["HOUSTON_*", "BAY_*", "GALV_*"],

    # North Hub
    "HB_NORTH": ["DFW_*", "WACO_*", "FTW_*"],

    # South Hub
    "HB_SOUTH": ["SANANT_*", "CORPUS_*", "LAREDO_*"],

    # West Hub
    "HB_WEST": ["ODESSA_*", "MIDLAND_*", "ABILENE_*"],

    # Panhandle
    "HB_PAN": ["AMARILLO_*", "LUBBOCK_*", "PANHANDLE_*"]
}
```

---

## 🔬 Key Challenges & Solutions

### Challenge 1: West Texas Negative Prices

**Problem:** West Texas frequently sees negative prices during high wind periods. Standard models don't handle this well.

**Solution:**
```python
# Output activation that allows negative prices
self.price_output = nn.Sequential(
    nn.Linear(hidden_dim, 24),
    nn.Tanh()  # Range: (-1, 1)
)

# Scale to realistic price range: -$50 to +$1000/MWh
price_forecast = price_output * price_scale + price_offset
# where price_scale = 525, price_offset = 475
# gives range: (-50, 1000)
```

**Features critical for negative price prediction:**
- West Texas wind forecast (high wind = negative prices)
- System load (low load + high wind = negative)
- Export constraint status
- Hour of day (middle of night = most negative)

### Challenge 2: Congestion Patterns

**Problem:** Congestion is not smooth - it's on/off based on binding constraints.

**Solution:** Two-stage model:
```python
# Stage 1: Predict if congestion will occur (classification)
congestion_binary = torch.sigmoid(self.classifier(features))  # P(congested)

# Stage 2: Predict magnitude if congested (regression)
congestion_magnitude = self.regressor(features)

# Combine
congestion_forecast = congestion_binary * congestion_magnitude
```

### Challenge 3: Data Sparsity for Small Nodes

**Problem:** Individual battery nodes may not have enough historical data.

**Solution:** Transfer learning
```python
# Train zone model on abundant data
zone_model.train(all_zone_data)

# Fine-tune on specific node with limited data
node_model = copy.deepcopy(zone_model)
node_model.fine_tune(node_specific_data, learning_rate=1e-5)
```

---

## 📈 Expected Performance by Zone

### Forecast Accuracy Targets

| Zone | DA MAE Target | RT MAE Target | Difficulty | Why? |
|------|---------------|---------------|------------|------|
| **HB_HOUSTON** | < $3/MWh | < $10/MWh | EASY | Stable, well-connected, follows system lambda |
| **HB_NORTH** | < $4/MWh | < $12/MWh | MEDIUM | Mix of wind + gas, moderate congestion |
| **HB_SOUTH** | < $5/MWh | < $15/MWh | MEDIUM | Transmission constraints, gas-heavy |
| **HB_WEST** | < $8/MWh | < $20/MWh | HARD | Extreme wind variability, negative prices, congestion |
| **HB_PAN** | < $7/MWh | < $18/MWh | HARD | Similar to West, remote from load |

### Why West Texas is Hardest

1. **Wind Forecast Error:** 1% wind error = $10+/MWh price error
2. **Transmission Limits:** Export capacity binding = instant price spike
3. **Negative Prices:** Difficult for models to predict
4. **Volatility:** Can swing from -$50 to +$200 in one hour

---

## 🎯 Implementation Plan

### Phase 1: System Lambda Model (Week 1)

- Train single system-wide lambda model
- Target: MAE < $5/MWh
- Use existing Model 1 architecture

### Phase 2: Houston Zone Model (Week 2)

- Start with easiest zone (Houston)
- Validate per-zone approach
- Target: MAE < $3/MWh (should beat system-wide)

### Phase 3: All Zone Models (Weeks 3-4)

- Train models for all 5 zones
- Validate each against historical data
- Compare to system-wide baseline

### Phase 4: Node-Specific Fine-Tuning (Week 5)

- Fine-tune for specific battery locations
- MOSS1, Gibbons Creek, Angleton, etc.
- Transfer learning from zone models

### Phase 5: Integration (Week 6)

- Integrate into shadow bidding system
- Replace system-wide forecast with per-node
- Validate: Do bids improve with node-specific prices?

---

## 📊 Features by Zone

### Houston Zone Features
```python
houston_features = [
    'houston_load',                # Local load
    'houston_temp',                # Local temperature
    'gas_price',                   # Henry Hub gas price
    'system_lambda',               # Base system cost
    'import_from_west',            # Wind imports
    'hour_of_day',
    'is_peak_hour',
]
```

### West Texas Features
```python
west_texas_features = [
    'west_wind_generation',        # LOCAL wind (CRITICAL)
    'west_wind_forecast_error',    # Forecast error drives congestion
    'export_capacity_available',   # Transmission limit
    'export_flow',                 # How much exporting now
    'system_load',                 # Need from rest of system
    'hour_of_day',                 # Wind patterns
    'season',                      # Summer = export constrained
    'negative_price_last_hour',    # Persistence (negative prices cluster)
]
```

**Key Difference:** West Texas model heavily weighted on **local wind**, Houston on **local load**.

---

## 🚀 Deployment Strategy

### For Each Battery

1. **Identify Node:** MOSS1 → HB_SOUTH zone
2. **Select Models:**
   - System lambda model
   - HB_SOUTH congestion model
   - HB_SOUTH loss model
   - MOSS1 node-specific adjustment

3. **Generate Forecast:**
   ```python
   moss1_price = (
       system_lambda_forecast +
       hb_south_congestion_forecast +
       hb_south_loss_forecast +
       moss1_node_adjustment
   )
   ```

4. **Use in Bidding:**
   - Bid to discharge when `moss1_price` > threshold
   - Bid to charge when `moss1_price` < threshold
   - More accurate node price → Better bidding decisions

---

## 💡 Key Insights

### Why This Matters

**Scenario: High Wind Day**

**System-Wide Model:**
- Predicts: Average price $40/MWh
- MOSS1 bids to discharge @ $40
- **Actual:** Houston $45 (good!), West Texas $10 (bad if we had battery there)

**Per-Node Model:**
- Houston model predicts: $45/MWh → Discharge (profit!)
- West Texas model predicts: $10/MWh → Don't discharge (avoid low price)
- **Result:** $35/MWh better decisions

**Revenue Impact:**
- 10 MW battery, 24 hours
- Bad decision: Discharge in West Texas @ $10 instead of Houston @ $45
- **Lost Revenue:** $35/MWh × 10 MW × 24 hrs = **$8,400 PER DAY**
- **Annual Impact:** $3.1M lost (if wrong zone every day)

### Critical for Multi-Battery Portfolio

If managing 5 batteries across Texas:
- Houston battery: Bid based on Houston forecast
- West battery: Bid based on West forecast
- North battery: Bid based on North forecast

**Coordinated bidding >> Independent bidding when you have location-specific forecasts.**

---

## 📚 Research to Implement

1. **"Locational Marginal Price Forecasting Using Deep Learning"** (IEEE TPS)
   - Node-specific price forecasting
   - Congestion pattern recognition

2. **"Transfer Learning for Price Forecasting in Multiple Markets"** (Applied Energy)
   - Train on abundant data, fine-tune on sparse
   - Applicable to zone → node transfer

3. **"Handling Negative Electricity Prices with Deep Learning"** (Energy Economics)
   - Special techniques for renewable-heavy zones
   - Critical for West Texas

---

**Bottom Line:** Per-node forecasting is ESSENTIAL for maximizing battery revenue. Location matters. West Texas ≠ Houston. This refinement could add **$1M+ annually** to portfolio revenue. 🚀
