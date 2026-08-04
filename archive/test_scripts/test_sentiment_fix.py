#!/usr/bin/env python3
"""
Test Sentiment Fallback Fix
Simulates the sentiment failure scenario to verify the fix works
"""

import sys
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_sentiment_fallback():
    """Test that sentiment failures are handled gracefully"""

    print("="*60)
    print("TESTING SENTIMENT FALLBACK FIX")
    print("="*60)
    print()

    # Import after adding to path
    from Trader_main_Grok4_20250731 import (
        load_configuration,
        engineer_features,
        add_sentiment_features
    )

    print("✓ Imports successful")

    # Load config
    config = load_configuration('config.yaml')
    print(f"✓ Config loaded (tickers: {config['tickers']})")

    # Create sample dataframe
    print("\nCreating sample data...")
    sample_data = {
        'ticker': ['AAPL'] * 10,
        'date': pd.date_range('2025-12-01', periods=10),
        'open': [150.0] * 10,
        'high': [152.0] * 10,
        'low': [149.0] * 10,
        'close': [151.0] * 10,
        'volume': [1000000] * 10
    }
    df = pd.DataFrame(sample_data)

    # Engineer features
    print("Engineering features...")
    df = engineer_features(df)

    if df.empty:
        print("❌ Feature engineering failed")
        return False

    print(f"✓ Features engineered: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Test sentiment with real API (this might fail)
    print("\nTesting sentiment with real API...")
    print("(This may take 10-30 seconds...)")

    try:
        df_with_sentiment = add_sentiment_features(df.copy(), config)

        if 'Sentiment_Score' in df_with_sentiment.columns:
            print("✓ Sentiment analysis succeeded!")
            print(f"  Sentiment scores: {df_with_sentiment['Sentiment_Score'].tolist()[:5]}")
            sentiment_worked = True
        else:
            print("⚠️  Sentiment column missing even after successful call")
            sentiment_worked = False

    except Exception as e:
        print(f"⚠️  Sentiment analysis failed with error: {e}")
        sentiment_worked = False

    # Test the fallback scenario by forcing a failure
    print("\nTesting fallback scenario (simulated failure)...")

    # Temporarily break the config to force failure
    bad_config = config.copy()
    bad_config['grok_api_key'] = 'invalid_key_to_force_failure'

    try:
        df_fallback = add_sentiment_features(df.copy(), bad_config)

        if 'Sentiment_Score' in df_fallback.columns:
            print("✓ Fallback worked! Sentiment_Score column exists")
            print(f"  Fallback values: {df_fallback['Sentiment_Score'].unique()}")

            if (df_fallback['Sentiment_Score'] == 0.0).all():
                print("✓ All sentiment scores are 0.0 (neutral fallback)")
                fallback_worked = True
            else:
                print("⚠️  Unexpected sentiment values in fallback")
                fallback_worked = False
        else:
            print("❌ FALLBACK FAILED - Sentiment_Score column missing!")
            fallback_worked = False

    except Exception as e:
        print(f"❌ FALLBACK FAILED - Exception raised: {e}")
        fallback_worked = False

    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    if sentiment_worked:
        print("✅ Real sentiment API: PASSED")
    else:
        print("⚠️  Real sentiment API: FAILED (expected if rate limited)")

    if fallback_worked:
        print("✅ Sentiment fallback: PASSED")
    else:
        print("❌ Sentiment fallback: FAILED")

    if fallback_worked:
        print("\n🎉 SUCCESS! The fix ensures trading continues even when sentiment fails.")
        print("\nWhat this means:")
        print("  • If sentiment API works → uses real sentiment data")
        print("  • If sentiment API fails → uses 0.0 (neutral) and continues")
        print("  • Trading is never blocked by sentiment failures")
        return True
    else:
        print("\n❌ FAILURE! The fallback is not working correctly.")
        print("\nThe fix may not have been applied correctly.")
        print("Check line 1230-1234 in Trader_main_Grok4_20250731.py")
        return False


if __name__ == '__main__':
    try:
        success = test_sentiment_fallback()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
