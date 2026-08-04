import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
from datetime import datetime

wb = Workbook()

# ============================================================
# Color palette
# ============================================================
DARK_BG = PatternFill('solid', fgColor='1B2A3D')
HEADER_BG = PatternFill('solid', fgColor='2C3E50')
ACCENT_BG = PatternFill('solid', fgColor='F39C12')
LIGHT_BG = PatternFill('solid', fgColor='ECF0F1')
WHITE_BG = PatternFill('solid', fgColor='FFFFFF')
GREEN_BG = PatternFill('solid', fgColor='E8F5E9')
RED_BG = PatternFill('solid', fgColor='FFEBEE')
YELLOW_BG = PatternFill('solid', fgColor='FFFDE7')
BLUE_INPUT = PatternFill('solid', fgColor='E3F2FD')

WHITE_FONT = Font(name='Arial', color='FFFFFF', bold=True, size=11)
HEADER_FONT = Font(name='Arial', color='FFFFFF', bold=True, size=10)
TITLE_FONT = Font(name='Arial', color='1B2A3D', bold=True, size=14)
SUBTITLE_FONT = Font(name='Arial', color='2C3E50', bold=True, size=11)
DATA_FONT = Font(name='Arial', color='2C3E50', size=10)
BLUE_FONT = Font(name='Arial', color='0000FF', size=10)  # hardcoded inputs
BLACK_FONT = Font(name='Arial', color='000000', size=10)  # formulas
GREEN_FONT = Font(name='Arial', color='008000', size=10)  # cross-sheet refs
ACCENT_FONT = Font(name='Arial', color='F39C12', bold=True, size=12)
SMALL_FONT = Font(name='Arial', color='7F8C8D', size=9)

thin_border = Border(
    left=Side(style='thin', color='BDC3C7'),
    right=Side(style='thin', color='BDC3C7'),
    top=Side(style='thin', color='BDC3C7'),
    bottom=Side(style='thin', color='BDC3C7')
)

center = Alignment(horizontal='center', vertical='center')
left_align = Alignment(horizontal='left', vertical='center')
right_align = Alignment(horizontal='right', vertical='center')

def style_header_row(ws, row, cols, fill=HEADER_BG, font=HEADER_FONT):
    for c in range(1, cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = fill
        cell.font = font
        cell.alignment = center
        cell.border = thin_border

def style_data_row(ws, row, cols, font=DATA_FONT, fill=WHITE_BG):
    for c in range(1, cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = font
        cell.fill = fill
        cell.border = thin_border
        cell.alignment = right_align if c > 1 else left_align

# ============================================================
# SHEET 1: Settings
# ============================================================
ws_settings = wb.active
ws_settings.title = 'Settings'
ws_settings.sheet_properties.tabColor = 'F39C12'

ws_settings['A1'] = 'Income Portfolio Tracker — Settings'
ws_settings['A1'].font = TITLE_FONT
ws_settings.merge_cells('A1:D1')

ws_settings['A3'] = 'Parameter'
ws_settings['B3'] = 'Value'
ws_settings['C3'] = 'Notes'
style_header_row(ws_settings, 3, 3)

settings_data = [
    ['Monthly Income Goal', 2000, 'Target passive income per month'],
    ['Annual Income Goal', '=B4*12', 'Calculated from monthly'],
    ['Portfolio Currency', 'USD', ''],
    ['Risk-Free Rate', 0.0441, '10yr Treasury yield — update weekly'],
    ['VIX Level', 31.0, 'CBOE VIX — update weekly'],
    ['2yr Treasury', 0.0396, 'Fed H.15 — update weekly'],
    ['10yr Treasury', 0.0441, 'Fed H.15 — update weekly'],
    ['HY OAS', 0.0317, 'ICE BofA HY OAS — update weekly'],
    ['Cash Reserve Min', 1500, 'Minimum cash to keep in money market'],
    ['Report Start Date', datetime(2026, 3, 30), 'First report date'],
    ['Owner', 'Shawn', ''],
]

for i, (param, val, note) in enumerate(settings_data, start=4):
    ws_settings.cell(row=i, column=1, value=param).font = DATA_FONT
    cell_b = ws_settings.cell(row=i, column=2, value=val)
    ws_settings.cell(row=i, column=3, value=note).font = SMALL_FONT
    if isinstance(val, str) and val.startswith('='):
        cell_b.font = BLACK_FONT
    elif isinstance(val, (int, float)):
        cell_b.font = BLUE_FONT
        cell_b.fill = BLUE_INPUT
    else:
        cell_b.font = DATA_FONT
    style_data_row(ws_settings, i, 3)

# Format percentages
for r in [7, 9, 10, 11]:
    ws_settings.cell(row=r, column=2).number_format = '0.00%'
ws_settings.cell(row=4, column=2).number_format = '$#,##0'
ws_settings.cell(row=5, column=2).number_format = '$#,##0'
ws_settings.cell(row=12, column=2).number_format = '$#,##0'

ws_settings.column_dimensions['A'].width = 25
ws_settings.column_dimensions['B'].width = 18
ws_settings.column_dimensions['C'].width = 40

# ============================================================
# SHEET 2: Allocation Targets
# ============================================================
ws_alloc = wb.create_sheet('Allocation_Targets')
ws_alloc.sheet_properties.tabColor = '3498DB'

ws_alloc['A1'] = 'Strategy Bucket Allocation Targets'
ws_alloc['A1'].font = TITLE_FONT
ws_alloc.merge_cells('A1:F1')

headers = ['Bucket', 'Target %', 'Description', 'Yield Range', 'Income Type', 'Example Tickers']
for c, h in enumerate(headers, 1):
    ws_alloc.cell(row=3, column=c, value=h)
style_header_row(ws_alloc, 3, 6)

buckets = [
    ['Income Core', 0.40, 'Options-income & preferred ETFs', '7-13%', 'Variable + Stable', 'JEPI, JEPQ, PFF, SPYI'],
    ['High-Yield Stability', 0.30, 'BDCs, REITs, high-yield dividend', '6-14%', 'Stable + Variable', 'ARCC, MAIN, O, MO, HTGC'],
    ['Income Anchors', 0.20, 'Broad dividend ETFs, blue chips', '2-4%', 'Stable', 'SCHD, VYM, HDV, XOM'],
    ['Growth', 0.10, 'Growth with small dividend', '0-2%', 'Cyclical', 'AAPL, MSFT, QQQ'],
]

for i, row_data in enumerate(buckets, start=4):
    for c, val in enumerate(row_data, 1):
        cell = ws_alloc.cell(row=i, column=c, value=val)
        cell.font = BLUE_FONT if c == 2 else DATA_FONT
        cell.fill = BLUE_INPUT if c == 2 else WHITE_BG
        cell.border = thin_border
        cell.alignment = center if c == 2 else left_align
    ws_alloc.cell(row=i, column=2).number_format = '0.0%'

# Total row
ws_alloc.cell(row=8, column=1, value='TOTAL').font = Font(name='Arial', bold=True, size=10)
ws_alloc.cell(row=8, column=2, value='=SUM(B4:B7)').font = BLACK_FONT
ws_alloc.cell(row=8, column=2).number_format = '0.0%'
style_data_row(ws_alloc, 8, 6, font=Font(name='Arial', bold=True, size=10))

ws_alloc.column_dimensions['A'].width = 22
ws_alloc.column_dimensions['B'].width = 12
ws_alloc.column_dimensions['C'].width = 35
ws_alloc.column_dimensions['D'].width = 14
ws_alloc.column_dimensions['E'].width = 18
ws_alloc.column_dimensions['F'].width = 30

# ============================================================
# SHEET 3: Holdings
# ============================================================
ws_hold = wb.create_sheet('Holdings')
ws_hold.sheet_properties.tabColor = '27AE60'

ws_hold['A1'] = 'Portfolio Holdings'
ws_hold['A1'].font = TITLE_FONT
ws_hold.merge_cells('A1:N1')
ws_hold['A2'] = 'Blue cells = manual input | Black cells = formulas | Update prices & yields weekly'
ws_hold['A2'].font = SMALL_FONT
ws_hold.merge_cells('A2:N2')

hold_headers = [
    'Ticker', 'Name', 'Bucket', 'Shares', 'Avg Cost', 'Current Price',
    'Market Value', 'Cost Basis', 'Unrealized G/L', 'G/L %',
    'Annual Yield %', 'Annual Income', 'Monthly Income', '% of Portfolio'
]
for c, h in enumerate(hold_headers, 1):
    ws_hold.cell(row=4, column=c, value=h)
style_header_row(ws_hold, 4, 14)

# Pre-populate 20 empty rows for holdings (rows 5-24)
for r in range(5, 25):
    # Input columns (blue): Ticker, Name, Bucket, Shares, Avg Cost, Current Price, Annual Yield %
    for c in [1, 2, 3, 4, 5, 6, 11]:
        cell = ws_hold.cell(row=r, column=c)
        cell.font = BLUE_FONT
        cell.fill = BLUE_INPUT
        cell.border = thin_border

    # Formula columns (black)
    row = r
    # Market Value = Shares * Current Price
    ws_hold.cell(row=row, column=7, value=f'=IF(D{row}="","",D{row}*F{row})').font = BLACK_FONT
    # Cost Basis = Shares * Avg Cost
    ws_hold.cell(row=row, column=8, value=f'=IF(D{row}="","",D{row}*E{row})').font = BLACK_FONT
    # Unrealized G/L = Market Value - Cost Basis
    ws_hold.cell(row=row, column=9, value=f'=IF(G{row}="","",G{row}-H{row})').font = BLACK_FONT
    # G/L % = Unrealized G/L / Cost Basis
    ws_hold.cell(row=row, column=10, value=f'=IF(H{row}="","",IF(H{row}=0,0,I{row}/H{row}))').font = BLACK_FONT
    # Annual Income = Market Value * Annual Yield %
    ws_hold.cell(row=row, column=12, value=f'=IF(G{row}="","",G{row}*K{row})').font = BLACK_FONT
    # Monthly Income = Annual Income / 12
    ws_hold.cell(row=row, column=13, value=f'=IF(L{row}="","",L{row}/12)').font = BLACK_FONT
    # % of Portfolio = Market Value / Total Market Value
    ws_hold.cell(row=row, column=14, value=f'=IF(G{row}="","",IF(G$26=0,0,G{row}/G$26))').font = BLACK_FONT

    for c in range(7, 15):
        ws_hold.cell(row=row, column=c).border = thin_border
        ws_hold.cell(row=row, column=c).alignment = right_align

    # Number formats
    ws_hold.cell(row=row, column=5).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=6).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=7).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=8).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=9).number_format = '$#,##0.00;($#,##0.00);"-"'
    ws_hold.cell(row=row, column=10).number_format = '0.0%'
    ws_hold.cell(row=row, column=11).number_format = '0.00%'
    ws_hold.cell(row=row, column=12).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=13).number_format = '$#,##0.00'
    ws_hold.cell(row=row, column=14).number_format = '0.0%'

# Cash row (row 25)
ws_hold.cell(row=25, column=1, value='SWVXX').font = BLUE_FONT
ws_hold.cell(row=25, column=2, value='Schwab Money Market').font = BLUE_FONT
ws_hold.cell(row=25, column=3, value='Cash').font = BLUE_FONT
ws_hold.cell(row=25, column=4, value='').font = BLUE_FONT
ws_hold.cell(row=25, column=5, value=1).font = BLUE_FONT
ws_hold.cell(row=25, column=6, value=0).font = BLUE_FONT  # Enter cash amount here
ws_hold.cell(row=25, column=7, value='=F25').font = BLACK_FONT
ws_hold.cell(row=25, column=11, value=0.042).font = BLUE_FONT
ws_hold.cell(row=25, column=12, value='=G25*K25').font = BLACK_FONT
ws_hold.cell(row=25, column=13, value='=L25/12').font = BLACK_FONT
for c in range(1, 15):
    ws_hold.cell(row=25, column=c).border = thin_border
    if c == 6:
        ws_hold.cell(row=25, column=c).fill = YELLOW_BG

# Totals row (row 26)
ws_hold.cell(row=26, column=1, value='TOTAL').font = Font(name='Arial', bold=True, size=10, color='1B2A3D')
ws_hold.cell(row=26, column=7, value='=SUM(G5:G25)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=8, value='=SUM(H5:H24)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=9, value='=SUM(I5:I24)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=10, value='=IF(H26=0,0,I26/H26)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=12, value='=SUM(L5:L25)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=13, value='=SUM(M5:M25)').font = Font(name='Arial', bold=True, color='000000', size=10)
ws_hold.cell(row=26, column=14, value='=IF(G26=0,0,SUM(G5:G24)/G26)').font = Font(name='Arial', bold=True, color='000000', size=10)

for c in range(1, 15):
    ws_hold.cell(row=26, column=c).border = thin_border
    ws_hold.cell(row=26, column=c).fill = PatternFill('solid', fgColor='D5DBDB')

ws_hold.cell(row=26, column=7).number_format = '$#,##0.00'
ws_hold.cell(row=26, column=8).number_format = '$#,##0.00'
ws_hold.cell(row=26, column=9).number_format = '$#,##0.00;($#,##0.00);"-"'
ws_hold.cell(row=26, column=10).number_format = '0.0%'
ws_hold.cell(row=26, column=12).number_format = '$#,##0.00'
ws_hold.cell(row=26, column=13).number_format = '$#,##0.00'
ws_hold.cell(row=26, column=14).number_format = '0.0%'

# Column widths
col_widths = [8, 25, 20, 8, 12, 14, 14, 14, 14, 10, 12, 14, 14, 12]
for i, w in enumerate(col_widths, 1):
    ws_hold.column_dimensions[get_column_letter(i)].width = w

# ============================================================
# SHEET 4: Dashboard
# ============================================================
ws_dash = wb.create_sheet('Dashboard')
ws_dash.sheet_properties.tabColor = 'E74C3C'

ws_dash['A1'] = 'Income Portfolio Dashboard'
ws_dash['A1'].font = TITLE_FONT
ws_dash.merge_cells('A1:F1')
ws_dash['A2'] = f'Generated: {datetime.now().strftime("%Y-%m-%d")} | Goal: $2,000/month'
ws_dash['A2'].font = SMALL_FONT
ws_dash.merge_cells('A2:F2')

# --- Portfolio Health Metrics ---
ws_dash['A4'] = 'Portfolio Health Metrics'
ws_dash['A4'].font = SUBTITLE_FONT

metrics = [
    ['Portfolio Value (invested)', "=Holdings!G26-Holdings!G25", '$#,##0'],
    ['Portfolio Value (incl. cash)', '=Holdings!G26', '$#,##0'],
    ['Cash', '=Holdings!G25', '$#,##0'],
    ['Monthly Income', '=Holdings!M26', '$#,##0.00'],
    ['Annual Income', '=Holdings!L26', '$#,##0.00'],
    ['Portfolio Yield', '=IF(B6=0,0,B10/B6)', '0.00%'],
    ['Monthly Goal', '=Settings!B4', '$#,##0'],
    ['Monthly Gap', '=B12-B9', '$#,##0.00;($#,##0.00);"-"'],
    ['Income Achievement', '=IF(B12=0,0,B9/B12)', '0.0%'],
]

ws_dash.cell(row=5, column=1, value='Metric').font = HEADER_FONT
ws_dash.cell(row=5, column=2, value='Value').font = HEADER_FONT
style_header_row(ws_dash, 5, 2)

for i, (label, formula, fmt) in enumerate(metrics, start=6):
    ws_dash.cell(row=i, column=1, value=label).font = DATA_FONT
    cell = ws_dash.cell(row=i, column=2, value=formula)
    cell.font = GREEN_FONT  # cross-sheet references
    cell.number_format = fmt
    style_data_row(ws_dash, i, 2)

# --- Allocation Drift ---
ws_dash['A17'] = 'Allocation & Drift'
ws_dash['A17'].font = SUBTITLE_FONT

drift_headers = ['Bucket', 'Actual %', 'Target %', 'Drift', 'Status']
for c, h in enumerate(drift_headers, 1):
    ws_dash.cell(row=18, column=c, value=h)
style_header_row(ws_dash, 18, 5)

bucket_names = ['Income Core', 'High-Yield Stability', 'Income Anchors', 'Growth']
for i, bname in enumerate(bucket_names, start=19):
    ws_dash.cell(row=i, column=1, value=bname).font = DATA_FONT
    # Actual % = SUMIF on Holdings bucket column / total invested
    ws_dash.cell(row=i, column=2, value=f'=IF(B$6=0,0,SUMPRODUCT((Holdings!C$5:Holdings!C$24=A{i})*Holdings!G$5:Holdings!G$24)/B$6)').font = GREEN_FONT
    # Target % from Allocation_Targets
    ws_dash.cell(row=i, column=3, value=f'=Allocation_Targets!B{i-15}').font = GREEN_FONT
    # Drift = Actual - Target
    ws_dash.cell(row=i, column=4, value=f'=B{i}-C{i}').font = BLACK_FONT
    # Status
    ws_dash.cell(row=i, column=5, value=f'=IF(ABS(D{i})<=0.03,"Within tolerance",IF(ABS(D{i})<=0.07,IF(D{i}>0,"OVERWEIGHT","UNDERWEIGHT"),IF(D{i}>0,"SEVERELY OVERWEIGHT","SEVERELY UNDERWEIGHT")))').font = BLACK_FONT

    for c in range(1, 6):
        ws_dash.cell(row=i, column=c).border = thin_border
    ws_dash.cell(row=i, column=2).number_format = '0.0%'
    ws_dash.cell(row=i, column=3).number_format = '0.0%'
    ws_dash.cell(row=i, column=4).number_format = '+0.0%;-0.0%;"-"'

# --- Income Progress ---
ws_dash['A25'] = 'Income Progress & Time-to-Goal'
ws_dash['A25'].font = SUBTITLE_FONT

ws_dash.cell(row=26, column=1, value='Scenario').font = HEADER_FONT
ws_dash.cell(row=26, column=2, value='Monthly Savings').font = HEADER_FONT
ws_dash.cell(row=26, column=3, value='Annual Deploy').font = HEADER_FONT
ws_dash.cell(row=26, column=4, value='Years (current yield)').font = HEADER_FONT
ws_dash.cell(row=26, column=5, value='Years (8% yield)').font = HEADER_FONT
style_header_row(ws_dash, 26, 5)

scenarios = [
    ['Low', 1500, 0.60],
    ['Mid', 2000, 0.70],
    ['High', 2500, 0.75],
    ['Aggressive', 3000, 0.80],
]

for i, (name, savings, rate) in enumerate(scenarios, start=27):
    ws_dash.cell(row=i, column=1, value=name).font = DATA_FONT
    ws_dash.cell(row=i, column=2, value=savings).font = BLUE_FONT
    ws_dash.cell(row=i, column=2).fill = BLUE_INPUT
    ws_dash.cell(row=i, column=2).number_format = '$#,##0'
    # Annual Deploy = Monthly * 12 * deploy rate
    ws_dash.cell(row=i, column=3, value=f'=B{i}*12*{rate}').font = BLACK_FONT
    ws_dash.cell(row=i, column=3).number_format = '$#,##0'
    # Required portfolio at current yield
    # Years = (Required - Current) / Annual Deploy
    # Required = (Goal * 12) / Yield
    ws_dash.cell(row=i, column=4, value=f'=IF(B$11=0,"N/A",IF(C{i}=0,"N/A",(((Settings!B5)/B$11)-B$6)/C{i}))').font = BLACK_FONT
    ws_dash.cell(row=i, column=4).number_format = '0.0'
    # Years at 8% yield
    ws_dash.cell(row=i, column=5, value=f'=IF(C{i}=0,"N/A",((Settings!B5/0.08)-B$6)/C{i})').font = BLACK_FONT
    ws_dash.cell(row=i, column=5).number_format = '0.0'
    for c in range(1, 6):
        ws_dash.cell(row=i, column=c).border = thin_border

# --- Concentration ---
ws_dash['A33'] = 'Income Concentration'
ws_dash['A33'].font = SUBTITLE_FONT

conc_headers = ['Metric', 'Value']
for c, h in enumerate(conc_headers, 1):
    ws_dash.cell(row=34, column=c, value=h)
style_header_row(ws_dash, 34, 2)

conc_metrics = [
    ['Top 1 Income Contributor %', '=IF(B9=0,0,LARGE(Holdings!L5:Holdings!L24,1)/B9)', '0.0%'],
    ['Top 3 Income Concentration %', '=IF(B9=0,0,(LARGE(Holdings!L5:Holdings!L24,1)+LARGE(Holdings!L5:Holdings!L24,2)+LARGE(Holdings!L5:Holdings!L24,3))/B9)', '0.0%'],
    ['Number of Positions', '=COUNTA(Holdings!A5:Holdings!A24)', '0'],
]

for i, (label, formula, fmt) in enumerate(conc_metrics, start=35):
    ws_dash.cell(row=i, column=1, value=label).font = DATA_FONT
    cell = ws_dash.cell(row=i, column=2, value=formula)
    cell.font = GREEN_FONT
    cell.number_format = fmt
    style_data_row(ws_dash, i, 2)

# --- Macro Signals ---
ws_dash['A40'] = 'Macro & Market Signals'
ws_dash['A40'].font = SUBTITLE_FONT

macro_headers = ['Signal', 'Value', 'Implication']
for c, h in enumerate(macro_headers, 1):
    ws_dash.cell(row=41, column=c, value=h)
style_header_row(ws_dash, 41, 3)

macro_rows = [
    ['2yr Treasury', '=Settings!B9', 'Rate cut expectations'],
    ['10yr Treasury', '=Settings!B10', 'Duration / yield curve'],
    ['VIX', '=Settings!B8', 'Volatility regime'],
    ['HY OAS', '=Settings!B11', 'Credit spread health'],
]

for i, (signal, formula, impl) in enumerate(macro_rows, start=42):
    ws_dash.cell(row=i, column=1, value=signal).font = DATA_FONT
    cell = ws_dash.cell(row=i, column=2, value=formula)
    cell.font = GREEN_FONT
    ws_dash.cell(row=i, column=3, value=impl).font = SMALL_FONT
    for c in range(1, 4):
        ws_dash.cell(row=i, column=c).border = thin_border
    if 'Treasury' in signal or signal == 'HY OAS':
        ws_dash.cell(row=i, column=2).number_format = '0.00%'
    else:
        ws_dash.cell(row=i, column=2).number_format = '0.0'

# Dashboard column widths
ws_dash.column_dimensions['A'].width = 30
ws_dash.column_dimensions['B'].width = 18
ws_dash.column_dimensions['C'].width = 18
ws_dash.column_dimensions['D'].width = 22
ws_dash.column_dimensions['E'].width = 20

# ============================================================
# SHEET 5: Weekly_Log
# ============================================================
ws_log = wb.create_sheet('Weekly_Log')
ws_log.sheet_properties.tabColor = '9B59B6'

ws_log['A1'] = 'Weekly Snapshot Log'
ws_log['A1'].font = TITLE_FONT
ws_log.merge_cells('A1:J1')
ws_log['A2'] = 'Add a row each week to track trends over time'
ws_log['A2'].font = SMALL_FONT

log_headers = ['Date', 'Portfolio Value', 'Cash', 'Monthly Income', 'Yield %',
               'Income Core %', 'HY Stability %', 'Anchors %', 'Growth %', 'Notes']
for c, h in enumerate(log_headers, 1):
    ws_log.cell(row=4, column=c, value=h)
style_header_row(ws_log, 4, 10)

# Pre-format 52 rows (one year of weekly entries)
for r in range(5, 57):
    ws_log.cell(row=r, column=1).number_format = 'YYYY-MM-DD'
    ws_log.cell(row=r, column=2).number_format = '$#,##0'
    ws_log.cell(row=r, column=3).number_format = '$#,##0'
    ws_log.cell(row=r, column=4).number_format = '$#,##0.00'
    ws_log.cell(row=r, column=5).number_format = '0.00%'
    for c in range(6, 10):
        ws_log.cell(row=r, column=c).number_format = '0.0%'
    for c in range(1, 11):
        ws_log.cell(row=r, column=c).border = thin_border
        ws_log.cell(row=r, column=c).font = BLUE_FONT
        ws_log.cell(row=r, column=c).fill = BLUE_INPUT

ws_log.column_dimensions['A'].width = 14
ws_log.column_dimensions['B'].width = 16
ws_log.column_dimensions['C'].width = 12
ws_log.column_dimensions['D'].width = 16
ws_log.column_dimensions['E'].width = 10
for col in 'FGHI':
    ws_log.column_dimensions[col].width = 16
ws_log.column_dimensions['J'].width = 35

# ============================================================
# SHEET 6: Watchlist
# ============================================================
ws_watch = wb.create_sheet('Watchlist')
ws_watch.sheet_properties.tabColor = '1ABC9C'

ws_watch['A1'] = 'Watchlist & Research'
ws_watch['A1'].font = TITLE_FONT
ws_watch.merge_cells('A1:G1')

watch_headers = ['Ticker', 'Name', 'Bucket', 'Yield %', 'Price', 'Notes', 'Action']
for c, h in enumerate(watch_headers, 1):
    ws_watch.cell(row=3, column=c, value=h)
style_header_row(ws_watch, 3, 7)

for r in range(4, 24):
    for c in range(1, 8):
        ws_watch.cell(row=r, column=c).font = BLUE_FONT
        ws_watch.cell(row=r, column=c).fill = BLUE_INPUT
        ws_watch.cell(row=r, column=c).border = thin_border
    ws_watch.cell(row=r, column=4).number_format = '0.00%'
    ws_watch.cell(row=r, column=5).number_format = '$#,##0.00'

ws_watch.column_dimensions['A'].width = 10
ws_watch.column_dimensions['B'].width = 28
ws_watch.column_dimensions['C'].width = 20
ws_watch.column_dimensions['D'].width = 10
ws_watch.column_dimensions['E'].width = 12
ws_watch.column_dimensions['F'].width = 40
ws_watch.column_dimensions['G'].width = 15

# ============================================================
# Set sheet order: Dashboard first
# ============================================================
wb.move_sheet('Dashboard', offset=-3)

# Save
output_path = '/sessions/exciting-inspiring-babbage/mnt/Trader_2025/income_portfolio/Income_Portfolio_Tracker.xlsx'
wb.save(output_path)
print(f'Saved to {output_path}')
