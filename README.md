**Market Price Prediction**
Predicts weekly retail market prices for 6 vegetables in Sri Lanka using a CatBoost model.

**Overview**
This component is part of a larger multi-model pipeline. It predicts next week's retail market prices for 6 vegetables by combining:

Historical price patterns (2010 → present)
Wholesale price predictions
Weather risk predictions (flood/drought)
Live fuel price data (scraped from CEYPETCO)
USD/LKR exchange rate

**Vegetables covered** : Bitter Gourd, Brinjals, Cabbage, Carrot, Pumpkin, Tomatoes

**Custom Week System:**
Week 1  → Jan 1  – Jan 7
Week 2  → Jan 8  – Jan 14
...
Week 52 → Dec 24 – Dec 31

