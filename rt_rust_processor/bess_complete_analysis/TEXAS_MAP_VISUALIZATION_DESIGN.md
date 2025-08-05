# Texas BESS & TBX Geographic Visualization Design

## Overview
Interactive map visualization showing BESS performance, TB2 revenue potential, and capture rates across Texas, with time-series animation capabilities.

## Core Components

### 1. Map Infrastructure

#### Base Map Configuration
```typescript
interface MapConfig {
  center: [29.5, -98.5]; // Texas center
  zoom: 6;
  bounds: {
    north: 36.5,
    south: 25.8,
    east: -93.5,
    west: -106.6
  };
  projection: 'EPSG:3857'; // Web Mercator
  baseLayers: {
    satellite: boolean;
    terrain: boolean;
    grid: boolean; // ERCOT grid overlay
  };
}
```

#### Data Layers
```typescript
interface MapLayers {
  // Settlement Points Layer
  settlementPoints: {
    type: 'circle';
    data: SettlementPointData[];
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'tb2Revenue'],
        0, 5,
        1000000, 30
      ];
      'circle-color': ['interpolate', ['linear'], ['get', 'captureRate'],
        0, '#ff0000',    // Red for poor capture
        0.5, '#ffff00',  // Yellow for moderate
        1.0, '#00ff00'   // Green for excellent
      ];
      'circle-opacity': 0.8;
    };
  };
  
  // BESS Locations Layer
  bessLocations: {
    type: 'symbol';
    data: BessLocationData[];
    layout: {
      'icon-image': 'battery-icon';
      'icon-size': ['interpolate', ['linear'], ['get', 'capacityMW'],
        10, 0.5,
        200, 2.0
      ];
    };
  };
  
  // Heat Map Layer
  revenueHeatmap: {
    type: 'heatmap';
    data: HeatmapData[];
    paint: {
      'heatmap-weight': ['get', 'tb2Revenue'];
      'heatmap-intensity': 1;
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0, 'rgba(33,102,172,0)',
        0.2, 'rgb(103,169,207)',
        0.4, 'rgb(209,229,240)',
        0.6, 'rgb(253,219,199)',
        0.8, 'rgb(239,138,98)',
        1, 'rgb(178,24,43)'
      ];
      'heatmap-radius': 50;
    };
  };
}
```

### 2. Time Series Animation

#### Timeline Controller
```typescript
interface TimelineController {
  startDate: Date;
  endDate: Date;
  currentDate: Date;
  playbackSpeed: number; // 1x, 2x, 5x
  interval: 'daily' | 'weekly' | 'monthly';
  
  controls: {
    play: () => void;
    pause: () => void;
    reset: () => void;
    seek: (date: Date) => void;
  };
  
  events: {
    onDateChange: (date: Date) => void;
    onPlaybackComplete: () => void;
  };
}
```

#### Animation Implementation
```typescript
class MapAnimator {
  private animationFrame: number;
  private isPlaying: boolean = false;
  
  play() {
    this.isPlaying = true;
    this.animate();
  }
  
  private animate() {
    if (!this.isPlaying) return;
    
    // Update current date
    this.currentDate = this.getNextDate();
    
    // Update map data
    this.updateMapData(this.currentDate);
    
    // Update UI elements
    this.updateTimeline();
    this.updateMetrics();
    
    // Schedule next frame
    this.animationFrame = requestAnimationFrame(() => {
      setTimeout(() => this.animate(), 1000 / this.playbackSpeed);
    });
  }
  
  private updateMapData(date: Date) {
    // Update circle sizes and colors based on date
    this.map.setPaintProperty('settlementPoints', 'circle-color', 
      this.getColorExpression(date));
    this.map.setPaintProperty('settlementPoints', 'circle-radius', 
      this.getRadiusExpression(date));
  }
}
```

### 3. Interactive Features

#### Node Popup Information
```typescript
interface NodePopup {
  settlementPoint: string;
  coordinates: [number, number];
  
  // Current Period Metrics
  currentMetrics: {
    date: Date;
    tb2Revenue: number;
    actualBessRevenue: number;
    captureRate: number;
    numberOfBess: number;
    avgBessSize: number;
  };
  
  // Historical Trends
  trends: {
    tb2Revenue: SparklineData;
    captureRate: SparklineData;
    priceVolatility: SparklineData;
  };
  
  // Top BESS at this node
  topPerformers: Array<{
    name: string;
    revenue: number;
    captureRate: number;
  }>;
}
```

#### Click and Hover Interactions
```typescript
map.on('click', 'settlementPoints', (e) => {
  const properties = e.features[0].properties;
  const popup = new mapboxgl.Popup()
    .setLngLat(e.lngLat)
    .setHTML(renderNodePopup(properties))
    .addTo(map);
});

map.on('mouseenter', 'settlementPoints', () => {
  map.getCanvas().style.cursor = 'pointer';
});

map.on('mouseleave', 'settlementPoints', () => {
  map.getCanvas().style.cursor = '';
});
```

### 4. Data Visualization Options

#### Map Modes
```typescript
enum MapMode {
  TB2_POTENTIAL = 'tb2_potential',        // Show TB2 revenue potential
  ACTUAL_REVENUE = 'actual_revenue',      // Show actual BESS revenue
  CAPTURE_RATE = 'capture_rate',          // Show % of TB2 captured
  PRICE_VOLATILITY = 'price_volatility',  // Show price volatility
  BESS_DENSITY = 'bess_density',          // Show BESS concentration
  REVENUE_GAP = 'revenue_gap'             // Show missed opportunities
}

interface MapModeConfig {
  mode: MapMode;
  colorScheme: ColorScale;
  legend: LegendConfig;
  metric: (data: NodeData) => number;
}
```

#### Color Scales
```typescript
const colorScales = {
  revenue: {
    name: 'Revenue Scale',
    colors: ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c'],
    domain: [0, 50000, 100000, 200000, 500000],
    unit: '$/MW-yr'
  },
  
  captureRate: {
    name: 'Capture Rate',
    colors: ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#4575b4'],
    domain: [0, 0.25, 0.5, 0.75, 1.0],
    unit: '%',
    formatter: (v: number) => `${(v * 100).toFixed(0)}%`
  },
  
  volatility: {
    name: 'Price Volatility',
    colors: ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
    domain: [0, 10, 20, 50, 100],
    unit: '$/MWh std dev'
  }
};
```

### 5. Performance Metrics Panel

#### Summary Statistics
```typescript
interface MapMetricsPanel {
  // Statewide Metrics
  statewide: {
    totalTB2Potential: number;
    totalBessRevenue: number;
    avgCaptureRate: number;
    topPerformingZone: string;
    worstPerformingZone: string;
  };
  
  // Time Period Comparison
  periodComparison: {
    current: MetricsSummary;
    previous: MetricsSummary;
    change: {
      tb2Potential: number;
      captureRate: number;
      bessCount: number;
    };
  };
  
  // Regional Breakdown
  regionalStats: Array<{
    region: string;
    tb2Potential: number;
    actualRevenue: number;
    captureRate: number;
    bessCount: number;
  }>;
}
```

### 6. Implementation with ECharts + Mapbox

#### Map Integration
```typescript
import mapboxgl from 'mapbox-gl';
import * as echarts from 'echarts';

class TexasBessMap {
  private map: mapboxgl.Map;
  private echartsOverlay: echarts.ECharts;
  
  constructor(container: HTMLElement) {
    // Initialize Mapbox
    this.map = new mapboxgl.Map({
      container,
      style: 'mapbox://styles/mapbox/light-v10',
      center: [-98.5, 29.5],
      zoom: 6
    });
    
    // Add ECharts overlay for advanced visualizations
    this.initEchartsOverlay();
  }
  
  private initEchartsOverlay() {
    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    this.map.getContainer().appendChild(canvas);
    
    this.echartsOverlay = echarts.init(canvas);
    
    // Sync with map movements
    this.map.on('move', () => this.syncEchartsOverlay());
  }
  
  addBessLayer(data: BessLocationData[]) {
    // Convert BESS data to GeoJSON
    const geojson = {
      type: 'FeatureCollection',
      features: data.map(bess => ({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: bess.coordinates
        },
        properties: {
          name: bess.resourceName,
          capacity: bess.capacityMW,
          revenue: bess.revenue,
          tb2Revenue: bess.tb2Revenue,
          captureRate: bess.revenue / bess.tb2Revenue
        }
      }))
    };
    
    this.map.addSource('bess-locations', {
      type: 'geojson',
      data: geojson
    });
    
    this.map.addLayer({
      id: 'bess-circles',
      type: 'circle',
      source: 'bess-locations',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['get', 'capacity'],
          10, 5,
          200, 20
        ],
        'circle-color': ['interpolate', ['linear'], ['get', 'captureRate'],
          0, '#ff0000',
          0.5, '#ffff00',
          1.0, '#00ff00'
        ],
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff'
      }
    });
  }
}
```

### 7. Data Pipeline

#### Real-time Data Updates
```typescript
interface MapDataPipeline {
  // Fetch latest data
  fetchLatestData: () => Promise<{
    settlements: SettlementPointData[];
    bess: BessLocationData[];
    tb2Calculations: Tb2ResultData[];
  }>;
  
  // Cache management
  cacheStrategy: {
    ttl: number; // Time to live in seconds
    maxSize: number; // Max cache size in MB
    evictionPolicy: 'LRU' | 'FIFO';
  };
  
  // WebSocket for real-time updates
  websocket: {
    url: string;
    events: {
      onPriceUpdate: (data: PriceUpdate) => void;
      onBessStatusChange: (data: BessStatus) => void;
    };
  };
}
```

## Usage Example

```typescript
// Initialize the map
const texasMap = new TexasBessMap('map-container');

// Load initial data
const data = await fetchMapData('2024-01-01');
texasMap.updateData(data);

// Set up time animation
const animator = new MapAnimator(texasMap, {
  startDate: new Date('2023-01-01'),
  endDate: new Date('2024-12-31'),
  interval: 'monthly',
  playbackSpeed: 2
});

// Add event listeners
texasMap.on('nodeClick', (node) => {
  showNodeDetails(node);
});

texasMap.on('bessClick', (bess) => {
  showBessPerformance(bess);
});

// Start animation
animator.play();
```

## Performance Considerations

1. **Data Aggregation**: Pre-aggregate data by zoom level
2. **Clustering**: Cluster nearby BESS at low zoom levels
3. **Progressive Loading**: Load details on demand
4. **WebGL Rendering**: Use Mapbox GL for smooth rendering
5. **Caching**: Cache calculated TB2 values