//ライブラリー
#include <Trade\Trade.mqh>

#import "MyLib.ex5"
   //決済注文のチェック
   void CheckForClose(CTrade *ExtTrade, int BuyExit, int SellExit);
   //新規注文のチェック
   void CheckForOpen(CTrade *ExtTrade, int BuyEntry, int SellEntry, double lot, double slpips, double tppips);
   //MQL4のiATR()関数と同等
   double iMyATR(string symbol, int timeframe, int period, int shift);
   //MQL4のiMA()関数と同等
   double iMyMA(string symbol, int timeframe, int period, int ma_shift, int ma_methid, int applied_price, int shift);
   //トレンド期間の計算
   double iTrendDuration(string symbol, int timeframe, int period, int shift);
   //ZScoreの計算
   double iZScore(string symbol, int timeframe, int period, int shift);
   //分を足に変換
   int MinuteToPeriod(int minute);
   //ポジションの選択
   bool SelectPosition(int magic);
#import