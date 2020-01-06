#property library

#include <MyLib.mqh>

#define EPS 0.00001

void email(int BuyEntry, int BuyExit, int SellEntry, int SellExit, string strategy) {
    bool res;
    static int BuyEntrySendMail = 0;
    static int BuyExitSendMail = 0;
    static int SellEntrySendMail = 0;
    static int SellExitSendMail = 0;

    // BuyExit
    if (BuyExit==1 && BuyExitSendMail==0 && BuyEntrySendMail==1) {
        res = SendMail(strategy, "Type:BuyExit\nSymol:" + Symbol() + "\nPrice:" + Bid);
        BuyExitSendMail = 1;
        BuyEntrySendMail = 0;
    }
    // SellExit
    if (SellExit==1 && SellExitSendMail==0 && SellEntrySendMail==1) {
        res = SendMail(strategy, "Type:SellExit\n" + "Symol:" + Symbol() + "\n" + "Price:" + Ask);
        SellExitSendMail = 1;
        SellEntrySendMail = 0;
    }
    // BuyEntry
    if (BuyEntry==1 && BuyEntrySendMail==0) {
        res = SendMail(strategy, "Type:BuyEntry\n" + "Symol:" + Symbol() + "\n" + "Price:" + Ask);
        BuyEntrySendMail = 1;
        BuyExitSendMail = 0;
    }
    // SellEntry
    if (SellEntry==1 && SellEntrySendMail==0) {
        res = SendMail(strategy, "Type:SellEntry\n" + "Symol:" + Symbol() + "\n" + "Price:" + Bid);
        SellEntrySendMail = 1;
        SellExitSendMail = 0;
    }
}

bool IsTradingTime(void) {
    bool ret;

    if (DayOfWeek()==6 || DayOfWeek()==0 || (DayOfWeek()==5 && Hour()>=17)){
        ret = False;
    }
    else {
        ret = True;
    }

    return ret;
}

double iHLBand(string symbol, int timeframe, string mode, int period, int shift) {
    double hband, lband, ret;
    int highest, lowest;

    highest = iHighest(symbol, timeframe, MODE_HIGH, period, shift);
    lowest = iLowest(symbol, timeframe, MODE_LOW, period, shift);
    if (mode == "high") {
        ret = iHigh(symbol, timeframe, highest);
    }
    else if (mode == "low") {
        ret = iLow(symbol, timeframe, lowest);
    }
    else {
        hband = iHigh(symbol, timeframe, highest);
        lband = iLow(symbol, timeframe, lowest);
        ret = (hband+lband) / 2.0;
    }

    return ret;
}

double iNoEntryLevel(string symbol, int timeframe, string type, int period, double pct, int shift) {
    double close, hband, lband;
    int ret;

    close = iClose(symbol, timeframe, shift);    
    if (type == "buy") {
        lband = iHLBand(symbol, timeframe, "low", period, shift);
        if (lband < EPS) {
            ret = 1;
        }
        else {
            if ((close-lband)/lband*100.0<pct) {
                ret = 1;
            }
            else {
                ret = 0;
            }
        }
    }
    else if (type == "sell") {
        hband = iHLBand(symbol, timeframe, "high", period, shift);
        if (close < EPS) {
            ret = 1;
        }
        else {
            if ((hband-close)/close*100.0<pct) {
                ret = 1;
            }
            else {
                ret = 0;
            }
        }
    }
    else {
        ret = 1;
    }

    return ret;
}

double iTrendDuration(string symbol, int timeframe, int period, int shift) {
    double high, low, ma, ret;
    int i;
    int down = 0;
    int up = 0;

    for (i=period*2+shift-1; i>=shift; i--) {
        high = iHigh(symbol, timeframe, i);
        low = iLow(symbol, timeframe, i);
        ma = iMA(symbol, timeframe, period, 0, MODE_SMA, PRICE_CLOSE, i);
        if (low > ma) {
            up += 1;
        }
        else {
            up = 0;
        }
        if (high < ma) {
            down += 1;
        }
        else {
            down = 0;
        }
    }
    ret = (up-down) / double(period);

    return ret;
}

double iZScore(string symbol, int timeframe, int period, int shift) {
    double close, ma, ret, std;

    close = iClose(symbol, timeframe, shift);
    ma = iMA(symbol, timeframe, period, 0, MODE_SMA, PRICE_CLOSE, shift);
    std = iStdDev(symbol, timeframe, period, 0, MODE_SMA, PRICE_CLOSE, shift);
    if (std < EPS) {
        ret = 0.0;
    }
    else {
        ret = (close-ma) / std;
    }
    return ret;
}

double ProfitParcentage(void) {
    static double balance_at_open = 0.0;
    double balance_now, profit_parcentage;

    if ((Hour()==0 && Minute()==0) || balance_at_open<EPS) {
        balance_at_open = AccountBalance();
    }
    balance_now = AccountBalance();
    if (MathAbs(balance_at_open) > EPS) {
        profit_parcentage = (balance_now-balance_at_open) / balance_at_open * 100.0;
    }
    else {
        profit_parcentage = 0.0;
    }

    return profit_parcentage;
}

int ToPeriod(int minute) {
    return minute / Period();
}

void trade(int BuyEntry, int BuyExit, int SellEntry, int SellExit, double lots, double slippage, int magic) {
    int cnt, pos, ticket, total;
    // exit
    total = OrdersTotal();
    for (cnt=0; cnt<total; cnt++) {
        if (!OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES)) {
            continue;
        }
        if (OrderSymbol()==Symbol() && OrderMagicNumber()==magic) {
            if (OrderType() == OP_BUY) {
                if (BuyExit == 1) {
                    if (!OrderClose(OrderTicket(), OrderLots(), Bid, slippage, Violet)) {
                        Print("OrderClose error ",GetLastError());
                    }
                }
            }
            if (OrderType() == OP_SELL) {
                if (SellExit == 1) {
                    if(!OrderClose(OrderTicket(), OrderLots(), Ask, slippage, Violet)) {
                        Print("OrderClose error ",GetLastError());
                    }
                }
            }
        }
    }
    // entry
    total = OrdersTotal();
    pos = 0;
    for (cnt=0; cnt<total; cnt++) {
        if (!OrderSelect(cnt, SELECT_BY_POS, MODE_TRADES)) {
            continue;
        }
        if (OrderSymbol()==Symbol() && OrderMagicNumber()==magic) {
            pos += 1;
        }
    }
    if (pos==0) {
        if (BuyEntry == 1) {
            ticket = OrderSend(Symbol(), OP_BUY, lots, Ask, slippage, 0, 0, "", magic, 0, Green);
        }
        else if (SellEntry == 1) {
            ticket = OrderSend(Symbol(), OP_SELL, lots, Bid, slippage, 0, 0, "", magic, 0, Red);
        }
    }
}
