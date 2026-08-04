#!/usr/bin/env python3
"""
Simple test to verify menu bar icon appears
"""
import sys
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import Qt

def create_icon(color_name):
    """Create a colored circle icon"""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    colors = {
        "red": QColor(244, 67, 54),
        "green": QColor(76, 175, 80),
        "yellow": QColor(255, 193, 7),
    }

    painter.setBrush(colors.get(color_name, colors["red"]))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()

    return QIcon(pixmap)

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Check if system tray is available
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("ERROR: System tray is not available on this system")
        return 1

    # Create tray icon
    tray = QSystemTrayIcon()
    tray.setIcon(create_icon("green"))
    tray.setToolTip("Test Trading Bot - If you can see this, the GUI works!")

    # Create simple menu
    menu = QMenu()

    test_action = QAction("✅ Menu Bar Icon Works!")
    test_action.setEnabled(False)
    menu.addAction(test_action)

    menu.addSeparator()

    quit_action = QAction("Quit Test")
    quit_action.triggered.connect(app.quit)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.show()

    print("")
    print("=" * 60)
    print("✅ Menu bar icon should now be visible!")
    print("=" * 60)
    print("")
    print("Look for a GREEN CIRCLE in your menu bar (top-right)")
    print("Click it and you should see: '✅ Menu Bar Icon Works!'")
    print("")
    print("If you CAN see it: The GUI system works!")
    print("If you CANNOT see it: There's a macOS display issue")
    print("")
    print("Press Ctrl+C or click 'Quit Test' in the menu to exit")
    print("=" * 60)

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
