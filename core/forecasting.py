"""
Forecasting Module for Claim Data Intelligence System
ML-based risk prediction using seasonality and historical trends
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


class RiskForecaster:
    """ML-based forecasting engine for proactive risk warnings"""
    
    def __init__(self):
        """Initialize forecasting engine"""
        self.model = None
        self.training_data = None
    
    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: str = "claim_date",
        value_column: str = "claim_amount",
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Prepare time series data for forecasting
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column to aggregate
            freq: Frequency for aggregation (D=daily, W=weekly, M=monthly)
            
        Returns:
            Time series DataFrame
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Aggregate by time period
        ts_data = df.groupby(pd.Grouper(key=date_column, freq=freq))[value_column].agg(['sum', 'count', 'mean'])
        ts_data.columns = ['total_amount', 'claim_count', 'avg_amount']
        ts_data = ts_data.reset_index()
        
        return ts_data
    
    def detect_seasonality(self, ts_data: pd.DataFrame, value_col: str = 'total_amount') -> Dict:
        """
        Detect seasonal patterns in the data
        
        Args:
            ts_data: Time series DataFrame
            value_col: Column to analyze for seasonality
            
        Returns:
            Dictionary with seasonality information
        """
        if len(ts_data) < 14:
            return {"has_seasonality": False, "period": None, "strength": 0}
        
        values = ts_data[value_col].values
        
        # Simple autocorrelation-based seasonality detection
        # Check for weekly (7-day) and 30-day patterns
        seasonal_info = {
            "has_seasonality": False,
            "period": None,
            "strength": 0,
            "patterns": {}
        }
        
        for period in [7, 30]:
            if len(values) > period * 2:
                autocorr = np.correlate(values, values, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if period < len(autocorr):
                    strength = autocorr[period] / autocorr[0] if autocorr[0] != 0 else 0
                    seasonal_info["patterns"][period] = float(strength)
                    
                    if strength > 0.5 and (seasonal_info["strength"] < strength):
                        seasonal_info["has_seasonality"] = True
                        seasonal_info["period"] = period
                        seasonal_info["strength"] = float(strength)
        
        return seasonal_info
    
    def simple_forecast(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        value_col: str = 'total_amount'
    ) -> pd.DataFrame:
        """
        Generate simple forecast using moving average and trend
        
        Args:
            ts_data: Time series DataFrame
            periods: Number of periods to forecast
            value_col: Column to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        if len(ts_data) < 7:
            # Not enough data, return simple average
            avg_value = ts_data[value_col].mean()
            last_date = ts_data.iloc[-1][ts_data.columns[0]]
            
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            return pd.DataFrame({
                'ds': forecast_dates,
                'yhat': [avg_value] * periods,
                'yhat_lower': [avg_value * 0.8] * periods,
                'yhat_upper': [avg_value * 1.2] * periods
            })
        
        # Calculate moving average and trend
        window = min(7, len(ts_data) // 2)
        ts_data['ma'] = ts_data[value_col].rolling(window=window).mean()
        
        # Simple linear trend
        x = np.arange(len(ts_data))
        y = ts_data[value_col].values
        
        # Handle NaN values
        valid_idx = ~np.isnan(y)
        if valid_idx.sum() > 1:
            coeffs = np.polyfit(x[valid_idx], y[valid_idx], 1)
            trend = coeffs[0]
        else:
            trend = 0
        
        # Generate forecast
        last_value = ts_data[value_col].iloc[-1]
        last_ma = ts_data['ma'].iloc[-1] if not pd.isna(ts_data['ma'].iloc[-1]) else last_value
        last_date = ts_data.iloc[-1][ts_data.columns[0]]
        
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecasts = []
        for i in range(periods):
            # Combine moving average with trend
            forecast_val = last_ma + trend * (i + 1)
            # Ensure non-negative (claim amounts cannot be negative)
            forecasts.append(max(0, forecast_val))
        
        # Add uncertainty bounds
        std_dev = ts_data[value_col].std()
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecasts,
            'yhat_lower': [max(0, f - 1.96 * std_dev) for f in forecasts],
            'yhat_upper': [f + 1.96 * std_dev for f in forecasts]
        })
        
        return forecast_df
    
    def calculate_risk_score(
        self,
        forecast: pd.DataFrame,
        historical_avg: float,
        threshold_multiplier: float = 1.5
    ) -> Dict:
        """
        Calculate risk score based on forecast
        
        Args:
            forecast: Forecast DataFrame
            historical_avg: Historical average value
            threshold_multiplier: Multiplier for risk threshold
            
        Returns:
            Dictionary with risk assessment
        """
        avg_forecast = forecast['yhat'].mean()
        max_forecast = forecast['yhat'].max()
        
        threshold = historical_avg * threshold_multiplier
        
        risk_level = "Low"
        risk_score = 0
        
        if max_forecast > threshold:
            risk_level = "High"
            risk_score = min(100, int((max_forecast / threshold - 1) * 100))
        elif avg_forecast > historical_avg * 1.2:
            risk_level = "Medium"
            risk_score = min(100, int((avg_forecast / historical_avg - 1) * 50))
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "avg_forecast": float(avg_forecast),
            "max_forecast": float(max_forecast),
            "historical_avg": float(historical_avg),
            "threshold": float(threshold)
        }
    
    def forecast_claims(
        self,
        df: pd.DataFrame,
        date_column: str = "claim_date",
        value_column: str = "claim_amount",
        forecast_days: int = 30
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Complete forecasting pipeline
        
        Args:
            df: Input claim data
            date_column: Name of date column
            value_column: Name of value column
            forecast_days: Number of days to forecast
            
        Returns:
            Tuple of (forecast DataFrame, seasonality info, risk assessment)
        """
        # Prepare time series
        ts_data = self.prepare_time_series(df, date_column, value_column, freq='D')
        
        # Detect seasonality
        seasonality = self.detect_seasonality(ts_data, 'total_amount')
        
        # Generate forecast
        forecast = self.simple_forecast(ts_data, forecast_days, 'total_amount')
        
        # Calculate risk
        historical_avg = ts_data['total_amount'].mean()
        risk_assessment = self.calculate_risk_score(forecast, historical_avg)
        
        return forecast, seasonality, risk_assessment
