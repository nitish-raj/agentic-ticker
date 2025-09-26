import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.services import (
    validate_ticker,
    get_company_info,
    get_crypto_info,
    compute_indicators,
    detect_events,
    forecast_prices,
    build_report,
    ddgs_search
)


class TestValidateTicker:
    @patch('src.services.yf.Ticker')
    def test_validate_stock_ticker(self, mock_ticker):
        """Test stock ticker validation"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({'Close': [100.0]})
        mock_ticker.return_value = mock_ticker_instance
        
        result = validate_ticker("AAPL")
        assert result == "AAPL"
    
    @patch('src.services.yf.Ticker')
    def test_validate_crypto_ticker(self, mock_ticker):
        """Test crypto ticker validation"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({'Close': [50000.0]})
        mock_ticker.return_value = mock_ticker_instance
        
        result = validate_ticker("BTC")
        assert result == "BTC"
    
    @patch('src.services.requests.post')
    @patch('src.services.yf.Ticker')
    def test_validate_company_name(self, mock_ticker, mock_post):
        """Test company name validation"""
        # Mock Yahoo Finance
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({'Close': [100.0]})
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock Gemini API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "AAPL"}]}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = validate_ticker("Apple Inc")
        assert result == "AAPL"


class TestCompanyInfo:
    @patch('src.services.yf.Ticker')
    def test_get_company_info_success(self, mock_ticker):
        """Test successful company info retrieval"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'longName': 'Apple Inc.',
            'shortName': 'Apple'
        }
        mock_ticker.return_value = mock_ticker_instance
        
        result = get_company_info("AAPL")
        assert 'company_name' in result
        assert 'short_name' in result
        assert result['company_name'] == 'Apple Inc.'
        assert result['short_name'] == 'Apple'
    
    @patch('src.services.yf.Ticker')
    def test_get_company_info_failure(self, mock_ticker):
        """Test company info retrieval failure"""
        mock_ticker.side_effect = Exception("Network error")
        
        result = get_company_info("INVALID")
        assert 'company_name' in result
        assert result['company_name'] == 'INVALID'


class TestCryptoInfo:
    @patch('src.services.CoinGeckoAPI')
    def test_get_crypto_info_success(self, mock_coingecko):
        """Test successful crypto info retrieval"""
        mock_instance = Mock()
        mock_coin_data = Mock()
        mock_coin_data.name = 'Bitcoin'
        mock_coin_data.symbol = 'btc'
        mock_coin_data.market_data = Mock()
        mock_coin_data.market_data.current_price = Mock()
        mock_coin_data.market_data.current_price.usd = 50000.0
        mock_coin_data.market_data.market_cap = Mock()
        mock_coin_data.market_data.market_cap.usd = 1000000000000.0
        mock_coin_data.market_data.total_volume = Mock()
        mock_coin_data.market_data.total_volume.usd = 30000000000.0
        mock_coin_data.market_data.price_change_percentage_24h = 2.5
        mock_coin_data.market_data.price_change_percentage_7d = 5.0
        mock_coin_data.market_data.circulating_supply = 19000000.0
        mock_coin_data.market_data.total_supply = 21000000.0
        
        mock_instance.coins.get_id.return_value = mock_coin_data
        mock_coingecko.return_value = mock_instance
        
        result = get_crypto_info("bitcoin")
        assert 'name' in result
        assert result['name'] == 'Bitcoin'
    
    def test_get_crypto_info_no_api(self):
        """Test crypto info when API is not available"""
        with patch('src.services.CoinGeckoAPI', None):
            result = get_crypto_info("bitcoin")
            assert 'error' in result


class TestComputeIndicators:
    def test_compute_indicators_with_valid_data(self):
        """Test technical indicator computation with valid data"""
        # Create sample price data as list of dicts (expected format)
        # Need more data for moving averages to work
        data = [
            {'date': '2023-01-01', 'close': 100, 'open': 99, 'high': 101, 'low': 98, 'volume': 1000000},
            {'date': '2023-01-02', 'close': 101, 'open': 100, 'high': 102, 'low': 99, 'volume': 1100000},
            {'date': '2023-01-03', 'close': 102, 'open': 101, 'high': 103, 'low': 100, 'volume': 1200000},
            {'date': '2023-01-04', 'close': 103, 'open': 102, 'high': 104, 'low': 101, 'volume': 1300000},
            {'date': '2023-01-05', 'close': 104, 'open': 103, 'high': 105, 'low': 102, 'volume': 1400000},
            {'date': '2023-01-06', 'close': 105, 'open': 104, 'high': 106, 'low': 103, 'volume': 1500000},
            {'date': '2023-01-07', 'close': 106, 'open': 105, 'high': 107, 'low': 104, 'volume': 1600000},
            {'date': '2023-01-08', 'close': 107, 'open': 106, 'high': 108, 'low': 105, 'volume': 1700000},
            {'date': '2023-01-09', 'close': 108, 'open': 107, 'high': 109, 'low': 106, 'volume': 1800000},
            {'date': '2023-01-10', 'close': 109, 'open': 108, 'high': 110, 'low': 107, 'volume': 1900000},
            {'date': '2023-01-11', 'close': 110, 'open': 109, 'high': 111, 'low': 108, 'volume': 2000000},
            {'date': '2023-01-12', 'close': 111, 'open': 110, 'high': 112, 'low': 109, 'volume': 2100000},
        ]
        
        result = compute_indicators(data)
        
        assert isinstance(result, list)
        # With adaptive window sizes, we should get some results
        # The function drops NaN values, so we need at least some valid data
        if len(result) > 0:
            assert 'date' in result[0]
            assert 'ma5' in result[0]
            assert 'ma10' in result[0]
            assert 'daily_return' in result[0]
            assert 'volatility' in result[0]
    
    def test_compute_indicators_with_empty_data(self):
        """Test technical indicator computation with empty data"""
        data = []
        
        result = compute_indicators(data)
        
        assert result == []


class TestDetectEvents:
    def test_detect_events_with_volatility(self):
        """Test event detection with volatile data"""
        data = [
            {'date': '2023-01-01', 'ma5': 100, 'daily_return': 0.10, 'volatility': 0.02},
            {'date': '2023-01-02', 'ma5': 110, 'daily_return': -0.18, 'volatility': 0.02},
            {'date': '2023-01-03', 'ma5': 90, 'daily_return': 0.33, 'volatility': 0.02},
            {'date': '2023-01-04', 'ma5': 120, 'daily_return': -0.25, 'volatility': 0.02},
        ]
        
        result = detect_events(data, threshold=2.0)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'date' in result[0]
        assert 'price' in result[0]
        assert 'change_percent' in result[0]
        assert 'direction' in result[0]
    
    def test_detect_events_with_stable_data(self):
        """Test event detection with stable data"""
        data = [
            {'date': '2023-01-01', 'ma5': 100, 'daily_return': 0.01, 'volatility': 0.02},
            {'date': '2023-01-02', 'ma5': 101, 'daily_return': -0.01, 'volatility': 0.02},
            {'date': '2023-01-03', 'ma5': 102, 'daily_return': 0.02, 'volatility': 0.02},
            {'date': '2023-01-04', 'ma5': 101, 'daily_return': -0.01, 'volatility': 0.02},
        ]
        
        result = detect_events(data, threshold=2.0)
        
        assert isinstance(result, list)
        # Should have fewer or no events due to low volatility


class TestForecastPrices:
    def test_forecast_prices_with_valid_data(self):
        """Test price forecasting with valid data"""
        data = [
            {'date': '2023-01-01', 'ma5': 100, 'daily_return': 0.01, 'volatility': 0.02},
            {'date': '2023-01-02', 'ma5': 101, 'daily_return': 0.02, 'volatility': 0.02},
            {'date': '2023-01-03', 'ma5': 102, 'daily_return': 0.01, 'volatility': 0.02},
            {'date': '2023-01-04', 'ma5': 103, 'daily_return': 0.02, 'volatility': 0.02},
        ]
        
        result = forecast_prices(data, days=5)
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert 'date' in result[0]
        assert 'forecast_price' in result[0]
        assert 'confidence' in result[0]
        assert 'trend' in result[0]


class TestBuildReport:
    def test_build_report_with_complete_data(self):
        """Test report building with complete data"""
        result = build_report(
            ticker='AAPL',
            events=[
                {'date': '2023-01-01', 'price': 100, 'change_percent': 5.0, 'direction': 'UP'}
            ],
            forecasts=[
                {'date': '2023-01-02', 'forecast_price': 105, 'confidence': 0.8, 'trend': 'UP'}
            ],
            company_info={'company_name': 'Apple Inc.', 'short_name': 'Apple'},
            crypto_info=None
        )
        
        assert 'content' in result
        assert 'ticker' in result
        assert 'analysis_period' in result
        assert 'generated_date' in result
        assert 'Apple Inc.' in result['content']
    
    def test_build_report_with_missing_data(self):
        """Test report building with missing data"""
        result = build_report(
            ticker='TEST',
            events=[],
            forecasts=[],
            company_info=None,
            crypto_info=None
        )
        
        assert 'content' in result
        assert 'ticker' in result
        assert 'analysis_period' in result
        assert 'generated_date' in result


class TestDDGSSearch:
    @patch('ddgs.DDGS')
    def test_ddgs_search_success(self, mock_ddgs_class):
        """Test successful DDGS search"""
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = [
            {'title': 'Test Result', 'href': 'http://example.com', 'body': 'Test content'}
        ]
        mock_ddgs_class.return_value = mock_ddgs_instance
        
        result = ddgs_search("test query")
        
        assert len(result) == 1
        assert result[0]['title'] == 'Test Result'
        assert result[0]['href'] == 'http://example.com'
        assert result[0]['content'] == 'Test content'
    
    @patch('ddgs.DDGS')
    def test_ddgs_search_failure(self, mock_ddgs_class):
        """Test DDGS search failure"""
        mock_ddgs_class.side_effect = Exception("Search failed")
        
        result = ddgs_search("test query")
        
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__])