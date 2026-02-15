#!/bin/bash

# סקריפט התקנה והפעלה למדריך Federated Learning

echo "=========================================="
echo "Federated Learning Tutorial - Setup"
echo "=========================================="
echo ""

# בדיקת Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 לא מותקן. אנא התקן Python 3.9 ומעלה"
    exit 1
fi

echo "✓ Python נמצא: $(python3 --version)"
echo ""

# יצירת סביבה וירטואלית
echo "📦 יוצר סביבה וירטואלית..."
python3 -m venv venv

# הפעלת הסביבה
echo "🔧 מפעיל סביבה וירטואלית..."
source venv/bin/activate

# התקנת תלויות
echo "📥 מתקין תלויות..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ ההתקנה הושלמה בהצלחה!"
echo "=========================================="
echo ""
echo "להפעלת המחברת:"
echo "1. הפעל: source venv/bin/activate"
echo "2. הפעל: jupyter notebook federated_learning_tutorial.ipynb"
echo ""
echo "או פשוט הרץ:"
echo "  ./run_notebook.sh"
echo ""
