#include <stderror.mqh>
#include <stdlib.mqh>

#import "mylib.ex4"

void email(int BuyEntry, int BuyExit, int SellEntry, int SellExit);
bool IsTradingTime(void);
double iHLBand(string symbol, int timeframe, string mode, int period, int shift);
double iNoEntryLevel(string symbol, int timeframe, string type, int period, double pct, int shift);
double iTrendDuration(string symbol, int timeframe, int period, int shift);
double iZScore(string symbol, int timeframe, int period, int shift);
double ProfitParcentage(void);
int ToPeriod(int minute);
void trade(int BuyEntry, int BuyExit, int SellEntry, int SellExit, double lots, double slippage, int magic);

#import