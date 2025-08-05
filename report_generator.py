#!/usr/bin/env python3
"""
Report generator for battery analysis results.
Creates Excel and HTML reports with visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate Excel and HTML reports for battery analysis."""
    
    def __init__(self):
        self.colors = {
            'HOUSTON': '#1f77b4',
            'NORTH': '#ff7f0e',
            'WEST': '#2ca02c',
            'AUSTIN': '#d62728',
            'SAN_ANTONIO': '#9467bd'
        }
        
    def generate_excel_report(self, results: Dict[str, pd.DataFrame], output_file: Path) -> None:
        """Generate Excel report with monthly and annual summaries."""
        logger.info("Generating Excel report")
        
        # Use pandas ExcelWriter with default engine
        with pd.ExcelWriter(output_file) as writer:
            # Create monthly summary
            monthly_df = self._prepare_monthly_summary(results)
            monthly_df.to_excel(writer, sheet_name='Monthly Summary')
            
            # Create annual summary
            annual_df = self._prepare_annual_summary(results)
            annual_df.to_excel(writer, sheet_name='Annual Summary')
            
            # Add daily details for each zone/battery combination
            for key, df in results.items():
                if len(df) > 0:
                    # Truncate sheet name if too long
                    sheet_name = key[:31]  # Excel limit is 31 chars
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
    def _prepare_monthly_summary(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare monthly summary data."""
        monthly_data = {}
        
        for key, df in results.items():
            if len(df) == 0:
                continue
                
            # Extract zone and battery type
            parts = key.split('_')
            zone = '_'.join(parts[:-1])
            battery_type = parts[-1]
            
            # Calculate monthly revenues
            df_copy = df.copy()
            df_copy['month'] = pd.to_datetime(df_copy['date']).dt.to_period('M')
            monthly = df_copy.groupby('month')['total_revenue'].sum()
            
            for month, revenue in monthly.items():
                month_str = str(month)
                if month_str not in monthly_data:
                    monthly_data[month_str] = {}
                monthly_data[month_str][f"{zone}_{battery_type}"] = revenue
        
        # Convert to DataFrame with proper column order
        monthly_df = pd.DataFrame(monthly_data).T
        
        # Reorder columns
        column_order = []
        for zone in ['HOUSTON', 'NORTH', 'WEST', 'AUSTIN', 'SAN_ANTONIO']:
            column_order.extend([f"{zone}_TB2", f"{zone}_TB4"])
        
        # Only include columns that exist
        existing_cols = [col for col in column_order if col in monthly_df.columns]
        monthly_df = monthly_df[existing_cols]
        monthly_df.index.name = 'Month'
        
        return monthly_df
        
    def _prepare_annual_summary(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare annual summary data."""
        annual_data = []
        
        for zone in ['HOUSTON', 'NORTH', 'WEST', 'AUSTIN', 'SAN_ANTONIO']:
            row_data = {'Zone': zone}
            
            # TB2 results
            key = f"{zone}_TB2"
            if key in results and len(results[key]) > 0:
                df = results[key]
                row_data['TB2 Revenue'] = df['total_revenue'].sum()
                row_data['TB2 Avg Daily'] = df['total_revenue'].mean()
                row_data['TB2 Cycles'] = df['cycles'].sum()
            else:
                row_data['TB2 Revenue'] = 0
                row_data['TB2 Avg Daily'] = 0
                row_data['TB2 Cycles'] = 0
                
            # TB4 results
            key = f"{zone}_TB4"
            if key in results and len(results[key]) > 0:
                df = results[key]
                row_data['TB4 Revenue'] = df['total_revenue'].sum()
                row_data['TB4 Avg Daily'] = df['total_revenue'].mean()
                row_data['TB4 Cycles'] = df['cycles'].sum()
            else:
                row_data['TB4 Revenue'] = 0
                row_data['TB4 Avg Daily'] = 0
                row_data['TB4 Cycles'] = 0
                
            annual_data.append(row_data)
            
        # Convert to DataFrame
        annual_df = pd.DataFrame(annual_data)
        
        return annual_df
        
            
    def generate_html_report(self, results: Dict[str, pd.DataFrame], output_file: Path) -> None:
        """Generate interactive HTML report with visualizations."""
        logger.info("Generating HTML report")
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, generating simple HTML report")
            self._generate_simple_html_report(results, output_file)
            return
        
        # Create figures
        figures = []
        
        # 1. Monthly revenue comparison
        fig_monthly = self._create_monthly_revenue_chart(results)
        figures.append(('Monthly Revenue Comparison', fig_monthly))
        
        # 2. Daily revenue time series
        fig_daily = self._create_daily_revenue_chart(results)
        figures.append(('Daily Revenue Time Series', fig_daily))
        
        # 3. Price spread analysis
        fig_spread = self._create_price_spread_chart(results)
        figures.append(('Price Spread Analysis', fig_spread))
        
        # 4. Zone performance comparison
        fig_zones = self._create_zone_comparison_chart(results)
        figures.append(('Zone Performance Comparison', fig_zones))
        
        # Generate HTML
        html_content = self._generate_html_template(figures, results)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
    def _create_monthly_revenue_chart(self, results: Dict[str, pd.DataFrame]):
        """Create monthly revenue comparison chart."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('TB2 Monthly Revenue', 'TB4 Monthly Revenue'),
            horizontal_spacing=0.1
        )
        
        for key, df in results.items():
            if len(df) == 0:
                continue
                
            parts = key.split('_')
            zone = '_'.join(parts[:-1])
            battery_type = parts[-1]
            
            # Calculate monthly revenues
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly = df.groupby('month')['total_revenue'].sum().reset_index()
            monthly['month'] = monthly['month'].astype(str)
            
            col = 1 if battery_type == 'TB2' else 2
            
            fig.add_trace(
                go.Bar(
                    x=monthly['month'],
                    y=monthly['total_revenue'],
                    name=zone,
                    marker_color=self.colors.get(zone, '#000000'),
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=2)
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        
        fig.update_layout(
            height=500,
            title_text="Monthly Battery Revenue by Zone",
            barmode='group'
        )
        
        return fig
        
    def _create_daily_revenue_chart(self, results: Dict[str, pd.DataFrame]):
        """Create daily revenue time series chart."""
        fig = go.Figure()
        
        for key, df in results.items():
            if len(df) == 0 or 'TB2' not in key:  # Only show TB2 for clarity
                continue
                
            parts = key.split('_')
            zone = '_'.join(parts[:-1])
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['total_revenue'],
                    name=zone,
                    line=dict(color=self.colors.get(zone, '#000000')),
                    mode='lines'
                )
            )
            
        fig.update_layout(
            title="Daily TB2 Revenue by Zone",
            xaxis_title="Date",
            yaxis_title="Daily Revenue ($)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_price_spread_chart(self, results: Dict[str, pd.DataFrame]):
        """Create price spread analysis chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average DA Price Spread', 'Average RT Price Spread'),
            vertical_spacing=0.15
        )
        
        spread_data = []
        
        for key, df in results.items():
            if len(df) == 0:
                continue
                
            parts = key.split('_')
            zone = '_'.join(parts[:-1])
            battery_type = parts[-1]
            
            spread_data.append({
                'Zone': zone,
                'Battery': battery_type,
                'DA Spread': df['da_price_spread'].mean(),
                'RT Spread': df['rt_price_spread'].mean()
            })
            
        spread_df = pd.DataFrame(spread_data)
        
        # DA spread
        for battery in ['TB2', 'TB4']:
            data = spread_df[spread_df['Battery'] == battery]
            fig.add_trace(
                go.Bar(
                    x=data['Zone'],
                    y=data['DA Spread'],
                    name=battery,
                    showlegend=True
                ),
                row=1, col=1
            )
            
        # RT spread
        for battery in ['TB2', 'TB4']:
            data = spread_df[spread_df['Battery'] == battery]
            fig.add_trace(
                go.Bar(
                    x=data['Zone'],
                    y=data['RT Spread'],
                    name=battery,
                    showlegend=False
                ),
                row=2, col=1
            )
            
        fig.update_yaxes(title_text="Price Spread ($/MWh)", row=1, col=1)
        fig.update_yaxes(title_text="Price Spread ($/MWh)", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="Average Price Spreads by Zone",
            barmode='group'
        )
        
        return fig
        
    def _create_zone_comparison_chart(self, results: Dict[str, pd.DataFrame]):
        """Create zone performance comparison chart."""
        # Calculate annual metrics
        comparison_data = []
        
        for zone in ['HOUSTON', 'NORTH', 'WEST', 'AUSTIN', 'SAN_ANTONIO']:
            for battery in ['TB2', 'TB4']:
                key = f"{zone}_{battery}"
                if key in results and len(results[key]) > 0:
                    df = results[key]
                    comparison_data.append({
                        'Zone': zone,
                        'Battery': battery,
                        'Annual Revenue': df['total_revenue'].sum(),
                        'Revenue per Cycle': df['total_revenue'].sum() / max(df['cycles'].sum(), 1)
                    })
                    
        comp_df = pd.DataFrame(comparison_data)
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Annual Revenue', 'Revenue per Cycle'),
            horizontal_spacing=0.15
        )
        
        # Annual revenue
        tb2_data = comp_df[comp_df['Battery'] == 'TB2']
        tb4_data = comp_df[comp_df['Battery'] == 'TB4']
        
        fig.add_trace(
            go.Bar(x=tb2_data['Zone'], y=tb2_data['Annual Revenue'], name='TB2'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=tb4_data['Zone'], y=tb4_data['Annual Revenue'], name='TB4'),
            row=1, col=1
        )
        
        # Revenue per cycle
        fig.add_trace(
            go.Bar(x=tb2_data['Zone'], y=tb2_data['Revenue per Cycle'], name='TB2', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=tb4_data['Zone'], y=tb4_data['Revenue per Cycle'], name='TB4', showlegend=False),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Annual Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Revenue per Cycle ($)", row=1, col=2)
        
        fig.update_layout(
            height=500,
            title_text="Zone Performance Comparison",
            barmode='group'
        )
        
        return fig
        
    def _generate_html_template(self, figures: List[tuple], results: Dict[str, pd.DataFrame]) -> str:
        """Generate HTML template with embedded charts."""
        # Calculate summary statistics
        summary_stats = []
        
        for zone in ['HOUSTON', 'NORTH', 'WEST', 'AUSTIN', 'SAN_ANTONIO']:
            for battery in ['TB2', 'TB4']:
                key = f"{zone}_{battery}"
                if key in results and len(results[key]) > 0:
                    df = results[key]
                    summary_stats.append({
                        'Zone': zone,
                        'Battery': battery,
                        'Annual Revenue': f"${df['total_revenue'].sum():,.2f}",
                        'Avg Daily Revenue': f"${df['total_revenue'].mean():,.2f}",
                        'Total Cycles': f"{df['cycles'].sum():,.1f}"
                    })
                    
        summary_df = pd.DataFrame(summary_stats)
        
        # Generate charts HTML
        charts_html = ""
        for title, fig in figures:
            charts_html += f'<div class="chart-container"><h2>{title}</h2>'
            charts_html += fig.to_html(include_plotlyjs=False, div_id=title.replace(' ', '_'))
            charts_html += '</div>'
            
        # Generate full HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ERCOT Battery Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary-table {{
            margin: 20px auto;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #366092;
            color: white;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .chart-container {{
            margin: 30px 0;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container h2 {{
            color: #366092;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>ERCOT Battery Energy Storage Analysis</h1>
    
    <h2 style="text-align: center;">Summary Statistics</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Zone</th>
                <th>Battery Type</th>
                <th>Annual Revenue</th>
                <th>Avg Daily Revenue</th>
                <th>Total Cycles</th>
            </tr>
        </thead>
        <tbody>
            {summary_df.to_html(index=False, header=False, escape=False)}
        </tbody>
    </table>
    
    {charts_html}
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_simple_html_report(self, results: Dict[str, pd.DataFrame], output_file: Path) -> None:
        """Generate simple HTML report without plotly."""
        # Calculate summary statistics
        summary_stats = []
        
        for zone in ['HOUSTON', 'NORTH', 'WEST', 'AUSTIN', 'SAN_ANTONIO']:
            for battery in ['TB2', 'TB4']:
                key = f"{zone}_{battery}"
                if key in results and len(results[key]) > 0:
                    df = results[key]
                    summary_stats.append({
                        'Zone': zone,
                        'Battery': battery,
                        'Annual Revenue': f"${df['total_revenue'].sum():,.2f}",
                        'Avg Daily Revenue': f"${df['total_revenue'].mean():,.2f}",
                        'Total Cycles': f"{df['cycles'].sum():,.1f}"
                    })
                    
        summary_df = pd.DataFrame(summary_stats)
        
        # Generate simple HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ERCOT Battery Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary-table {{
            margin: 20px auto;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #366092;
            color: white;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>ERCOT Battery Energy Storage Analysis</h1>
    
    <h2 style="text-align: center;">Summary Statistics</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Zone</th>
                <th>Battery Type</th>
                <th>Annual Revenue</th>
                <th>Avg Daily Revenue</th>
                <th>Total Cycles</th>
            </tr>
        </thead>
        <tbody>
            {summary_df.to_html(index=False, header=False, escape=False)}
        </tbody>
    </table>
    
    <p style="text-align: center; color: #666;">
        Note: Interactive charts require plotly installation. Install with: pip install plotly
    </p>
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)