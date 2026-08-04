#!/usr/bin/env python3
"""
Quick Fix: Add Sentiment Fallback
Patches Trader_main_Grok4_20250731.py to handle sentiment failures gracefully
"""

import sys
from pathlib import Path

def apply_fix():
    """Apply sentiment fallback fix to main trading bot"""

    bot_file = Path('Trader_main_Grok4_20250731.py')

    if not bot_file.exists():
        print(f"❌ Error: {bot_file} not found")
        return False

    print(f"📖 Reading {bot_file}...")
    with open(bot_file, 'r') as f:
        lines = f.readlines()

    # Find the line to fix
    target_line = None
    for i, line in enumerate(lines):
        if 'engineered_data = add_sentiment_features(engineered_data, config)' in line:
            # Check if it's in the live trading section (around line 1230)
            if i > 1200 and i < 1250:
                target_line = i
                break

    if target_line is None:
        print("❌ Could not find sentiment call in live trading section")
        return False

    print(f"✓ Found sentiment call at line {target_line + 1}")

    # Check if already fixed
    if 'try:' in lines[target_line - 1]:
        print("✓ Fix already applied! No changes needed.")
        return True

    # Backup original file
    backup_file = Path('Trader_main_Grok4_20250731.py.backup')
    print(f"💾 Creating backup at {backup_file}...")
    with open(backup_file, 'w') as f:
        f.writelines(lines)

    # Get the indentation of the current line
    indent = len(lines[target_line]) - len(lines[target_line].lstrip())
    indent_str = ' ' * indent

    # Create the fixed version
    fixed_lines = [
        f"{indent_str}try:\n",
        f"{indent_str}    engineered_data = add_sentiment_features(engineered_data, config)\n",
        f"{indent_str}except Exception as e:\n",
        f"{indent_str}    logger.error(f\"Sentiment analysis failed: {{e}}. Using zero sentiment as fallback.\")\n",
        f"{indent_str}    engineered_data['Sentiment_Score'] = 0.0\n",
    ]

    # Replace the original line with the fixed version
    lines[target_line:target_line+1] = fixed_lines

    # Write the fixed file
    print(f"✏️  Applying fix...")
    with open(bot_file, 'w') as f:
        f.writelines(lines)

    print("✅ Fix applied successfully!")
    print()
    print("Changes made:")
    print("-" * 60)
    print("Before:")
    print("    engineered_data = add_sentiment_features(engineered_data, config)")
    print()
    print("After:")
    print("    try:")
    print("        engineered_data = add_sentiment_features(engineered_data, config)")
    print("    except Exception as e:")
    print("        logger.error(f\"Sentiment analysis failed: {e}. Using zero sentiment as fallback.\")")
    print("        engineered_data['Sentiment_Score'] = 0.0")
    print("-" * 60)
    print()
    print("Next steps:")
    print("  1. Review the changes in Trader_main_Grok4_20250731.py")
    print("  2. Restart the bot service: ./1_start_bot_service.sh")
    print("  3. Monitor tomorrow's 3:55 PM execution")
    print()
    print(f"Backup saved at: {backup_file}")
    print("(You can restore with: cp Trader_main_Grok4_20250731.py.backup Trader_main_Grok4_20250731.py)")

    return True


if __name__ == '__main__':
    print("="*60)
    print("SENTIMENT FALLBACK FIX")
    print("="*60)
    print()
    print("This fix ensures trading continues even if sentiment analysis fails.")
    print()

    success = apply_fix()

    sys.exit(0 if success else 1)
