#!/bin/bash

# סקריפט להפעלת המחברת

echo "🚀 מפעיל את מחברת Federated Learning..."
echo ""

# בדיקה אם הסביבה קיימת
if [ ! -d "venv" ]; then
    echo "❌ סביבה וירטואלית לא נמצאה"
    echo "הרץ תחילה: ./setup.sh"
    exit 1
fi

# הפעלת הסביבה
source venv/bin/activate

# הפעלת Jupyter
jupyter notebook federated_learning_tutorial.ipynb
