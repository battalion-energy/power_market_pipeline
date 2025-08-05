# TBX Performance Analysis & Benchmarking Design

## Overview
Comprehensive system for comparing actual BESS performance against theoretical TB2 (Top-Bottom 2 hours) revenue potential, identifying gaps, and visualizing opportunities across Texas.

## Core Metrics

### 1. TBX Capture Rate
The fundamental metric comparing actual BESS revenue to theoretical TB2 potential.

```typescript
interface TbxCaptureMetrics {
  // Primary Metrics
  actualRevenue: number;           // Actual BESS revenue achieved
  tb2Potential: number;           // Theoretical TB2 revenue at node
  captureRate: number;            // actualRevenue / tb2Potential
  
  // Detailed Breakdown
  energyCapture: {
    actual: number;
    tb2Potential: number;
    rate: number;
  };
  
  ancillaryCapture: {
    actual: number;
    opportunity: number;         // AS revenue that doesn't conflict with TB2
    rate: number;
  };
  
  // Performance Classification
  tier: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  percentileRank: number;       // Where this BESS ranks in capture rate
}
```

### 2. Missed Opportunity Analysis

```typescript
interface MissedOpportunity {
  timestamp: Date;
  type: 'Charge' | 'Discharge';
  
  // What should have happened (TB2)
  tb2Action: {
    mw: number;
    price: number;
    revenue: number;
  };
  
  // What actually happened
  actualAction: {
    mw: number;
    price: number;
    revenue: number;
  };
  
  // The gap
  missedRevenue: number;
  
  // Why it was missed
  reason: MissedOpportunityReason;
}

enum MissedOpportunityReason {
  NOT_CHARGED = 'Not charged when cheap prices available',
  NOT_DISCHARGED = 'Not discharged when high prices available',
  WRONG_TIMING = 'Charged/discharged at suboptimal times',
  AS_COMMITMENT = 'Committed to ancillary services',
  MAINTENANCE = 'Unit on maintenance',
  OPERATIONAL_CONSTRAINT = 'Other operational constraint',
  UNKNOWN = 'Unknown reason'
}
```

### 3. Geographic Performance Patterns

```typescript
interface GeographicPerformance {
  zone: string;
  avgTb2Potential: number;
  avgActualRevenue: number;
  avgCaptureRate: number;
  
  // Location-specific factors
  priceVolatility: number;      // Standard deviation of prices
  congestionFrequency: number;  // % of hours with congestion
  renewablePenetration: number; // % renewable generation nearby
  
  // Performance distribution
  captureRateDistribution: {
    excellent: number;  // % of BESS >90% capture
    good: number;       // % of BESS 70-90% capture
    fair: number;       // % of BESS 50-70% capture
    poor: number;       // % of BESS <50% capture
  };
}
```

## Analysis Components

### 1. Daily Performance Tracking

```typescript
interface DailyTbxAnalysis {
  date: Date;
  resourceName: string;
  
  // TB2 Calculation
  tb2Windows: {
    charge: TimeWindow[];
    discharge: TimeWindow[];
    theoreticalRevenue: number;
  };
  
  // Actual Performance
  actualOperations: {
    charges: ActualOperation[];
    discharges: ActualOperation[];
    totalRevenue: number;
  };
  
  // Comparison
  captureRate: number;
  missedWindows: MissedOpportunity[];
  
  // Operational metrics
  cyclesAchieved: number;
  avgStateOfCharge: number;
  utilizationRate: number;
}
```

### 2. Rolling Performance Windows

```typescript
interface RollingPerformance {
  // Different time windows
  daily: TbxCaptureMetrics;
  weekly: TbxCaptureMetrics;
  monthly: TbxCaptureMetrics;
  quarterly: TbxCaptureMetrics;
  annual: TbxCaptureMetrics;
  
  // Trends
  trend: {
    improving: boolean;
    rateOfChange: number; // % per month
    projection: number;   // Expected capture rate in 3 months
  };
  
  // Consistency
  volatility: number;     // Standard deviation of daily capture rates
  bestDay: Date;
  worstDay: Date;
}
```

### 3. Peer Comparison Framework

```typescript
interface PeerComparison {
  resource: string;
  
  // Peer groups
  bySize: {
    peerGroup: string;  // e.g., "50-100 MW"
    rank: number;
    outOf: number;
    percentile: number;
  };
  
  byLocation: {
    zone: string;
    rank: number;
    outOf: number;
    percentile: number;
  };
  
  byDuration: {
    durationGroup: string; // e.g., "2-hour batteries"
    rank: number;
    outOf: number;
    percentile: number;
  };
  
  byOperator: {
    operator: string;
    rankWithinOperator: number;
    operatorAvgCapture: number;
  };
}
```

## Visualization Components

### 1. Capture Rate Timeline
Shows how capture rate evolves over time with overlays for market events.

```typescript
interface CaptureRateTimeline {
  type: 'line';
  xAxis: Date[];
  series: [
    {
      name: 'Capture Rate';
      data: number[];
      markLine: {
        data: [
          { name: 'Target', yAxis: 80 },
          { name: 'Fleet Average', yAxis: 65 }
        ];
      };
      markArea: {
        data: [
          // Highlight maintenance periods, market events, etc.
        ];
      };
    }
  ];
}
```

### 2. Opportunity Heatmap
24x7 grid showing when opportunities are typically missed.

```typescript
interface OpportunityHeatmap {
  type: 'heatmap';
  data: Array<{
    hour: number;      // 0-23
    dayOfWeek: number; // 0-6
    missedValue: number;
    missedCount: number;
  }>;
  
  tooltip: {
    formatter: (params) => {
      return `${params.dayOfWeek} ${params.hour}:00
              Missed: $${params.missedValue}
              Count: ${params.missedCount} times`;
    };
  };
}
```

### 3. Geographic Performance Map
Interactive Texas map with TB2 and capture rate layers.

```typescript
interface PerformanceMapConfig {
  layers: [
    {
      id: 'tb2-potential';
      type: 'choropleth';
      data: ZonalTb2Data[];
      colorScale: 'Blues';
      opacity: 0.7;
    },
    {
      id: 'bess-performance';
      type: 'bubble';
      data: BessLocationData[];
      sizeBy: 'capacity';
      colorBy: 'captureRate';
      colorScale: ['red', 'yellow', 'green'];
    },
    {
      id: 'missed-opportunities';
      type: 'heatmap';
      data: MissedOpportunityDensity[];
      radius: 50;
      blur: 15;
    }
  ];
  
  timeline: {
    enabled: true;
    autoPlay: false;
    data: DateRange;
    playInterval: 1000;
  };
}
```

## Implementation Architecture

### 1. Data Pipeline

```typescript
class TbxPerformanceAnalyzer {
  constructor(
    private tbxCalculator: TbxCalculator,
    private actualDataSource: BessDataSource,
    private geoService: GeographicService
  ) {}
  
  async analyzePerformance(
    resource: string,
    dateRange: DateRange
  ): Promise<PerformanceAnalysis> {
    // 1. Calculate theoretical TB2
    const tb2Results = await this.tbxCalculator.calculate(
      resource,
      dateRange
    );
    
    // 2. Get actual performance
    const actualResults = await this.actualDataSource.getRevenue(
      resource,
      dateRange
    );
    
    // 3. Compare and analyze
    const analysis = this.comparePerformance(tb2Results, actualResults);
    
    // 4. Identify missed opportunities
    const missed = this.identifyMissedOpportunities(
      tb2Results,
      actualResults
    );
    
    // 5. Calculate metrics
    return {
      captureRate: analysis.captureRate,
      missedOpportunities: missed,
      performanceTier: this.classifyPerformance(analysis.captureRate),
      recommendations: this.generateRecommendations(analysis, missed)
    };
  }
}
```

### 2. Real-time Monitoring

```typescript
interface RealtimeMonitoring {
  // WebSocket connection for live data
  websocket: {
    url: 'wss://ercot-stream.example.com';
    subscriptions: ['prices', 'bess-dispatch'];
  };
  
  // Real-time TB2 calculation
  liveTb2: {
    updateInterval: 300; // seconds
    rollingWindow: 24;  // hours
    alerts: {
      missedOpportunity: {
        threshold: 1000; // $ threshold
        notification: 'email' | 'sms' | 'webhook';
      };
    };
  };
  
  // Dashboard updates
  dashboard: {
    refreshRate: 60; // seconds
    metrics: ['captureRate', 'dailyRevenue', 'missedOpportunities'];
  };
}
```

### 3. Reporting Framework

```typescript
interface TbxPerformanceReport {
  // Executive Summary
  summary: {
    period: DateRange;
    fleetAverageCaptureRate: number;
    totalMissedRevenue: number;
    topPerformers: ResourceRanking[];
    bottomPerformers: ResourceRanking[];
  };
  
  // Detailed Analysis
  byResource: Map<string, {
    captureRate: number;
    actualRevenue: number;
    tb2Potential: number;
    missedRevenue: number;
    topMissedOpportunities: MissedOpportunity[];
  }>;
  
  // Geographic Analysis
  byZone: Map<string, GeographicPerformance>;
  
  // Trends
  trends: {
    captureRateImproving: boolean;
    monthlyTrend: TrendData[];
    projectedAnnualMissed: number;
  };
  
  // Recommendations
  recommendations: {
    immediate: string[];  // Quick wins
    shortTerm: string[];  // 1-3 months
    longTerm: string[];   // 3+ months
  };
}
```

## Key Performance Indicators (KPIs)

### Fleet-Level KPIs
1. **Average TB2 Capture Rate**: Fleet-wide average
2. **Total Missed Revenue**: Sum of all missed opportunities
3. **Capture Rate Trend**: Month-over-month improvement
4. **Geographic Efficiency**: Best/worst performing zones

### Resource-Level KPIs
1. **Individual Capture Rate**: vs TB2 potential
2. **Peer Ranking**: Within size/location/duration group
3. **Consistency Score**: Standard deviation of daily rates
4. **Improvement Trend**: Personal best tracking

### Operational KPIs
1. **Response Time**: How quickly BESS responds to price signals
2. **Forecast Accuracy**: Predicted vs actual price spreads
3. **AS Conflict Rate**: % of TB2 opportunities missed due to AS
4. **Technical Availability**: % time available for dispatch

## Integration Points

### 1. With TBX Calculator
- Real-time TB2 calculations
- Historical TB2 analysis
- What-if scenarios

### 2. With BESS Operations
- Actual dispatch data
- State of charge tracking
- Maintenance schedules

### 3. With Market Data
- Price feeds
- Congestion patterns
- System conditions

### 4. With Visualization Platform
- Real-time dashboards
- Geographic displays
- Performance reports