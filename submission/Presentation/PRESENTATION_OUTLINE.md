# Final Presentation Structure (15-20 minutes)
## Group 5: Multi-Modal Stock Prediction

### Slide 1: Title (30 sec)
- Project title
- Group members
- Target: SPY prediction with <15% MAPE

### Slide 2: Problem Statement (1 min)
- Challenge: Stock market prediction is complex
- Goal: Combine technical analysis + news sentiment
- Target: Sub-15% MAPE

### Slide 3: Data Overview (2 min)
- **SPY (S&P 500 ETF):** Feb 2024 - Nov 2025 (452 days)
- **Technical:** 46 indicators from 5 market indices
- **Sentiment:** 25,639 articles on 7 tech giants (FinBERT analysis)
- Show data timeline visualization

### Slide 4: Architecture (2 min)
- PyTorch LSTM: 2 layers, 128 hidden units
- Input: 48 features × 30-day sequences
- 228,993 parameters
- Show architecture diagram

### Slide 5: Methodology (2 min)
- Phase 1: Baseline LSTM (technical only)
- Phase 2: Dataset expansion (133→501 days)
- Phase 3: Sentiment integration (FinBERT)
- 70/15/15 train/val/test split

### Slide 6: Results - The Journey (3 min) ⭐
- **Baseline:** 52.31% MAPE
- **2-Year Data:** 18.03% MAPE (65.5% improvement)
- **+ Sentiment:** **7.87% MAPE** (85% total improvement)
- Show performance comparison bar chart
- **Key insight:** Multi-modal > single-source

### Slide 7: Key Learnings (2 min)
1. Dataset size is critical (3.8× data → 65.5% boost)
2. Sentiment matters (news adds 56.3% improvement)
3. Feature engineering crucial (46 technical features)
4. Sequence optimization (30-day window optimal)

### Slide 8: Technical Challenges (2 min)
- GitHub LFS issues with 1.4GB files
- FinBERT processing 4.6M articles
- Sentiment-stock alignment complexity
- AWS GPU infrastructure management

### Slide 9: Live Demo (2-3 min) 🚀
**Option A:** Show prediction on recent data
**Option B:** Show feature importance analysis
**Option C:** Compare prediction vs actual (test set)

### Slide 10: Future Work (1 min)
- Real-time deployment
- Additional sentiment sources (Twitter, Reddit)
- Multi-stock portfolio optimization
- Transfer learning to other indices

### Slide 11: Conclusion (1 min)
- ✅ Exceeded 15% MAPE target (achieved 7.87%)
- ✅ Demonstrated multi-modal learning effectiveness
- ✅ Production-ready system
- Thank you + Questions

### Backup Slides:
- Detailed feature list
- Training curves
- Error distribution analysis
- Code repository structure
