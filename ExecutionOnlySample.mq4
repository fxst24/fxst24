// ファイル名は31文字以内にすること。
extern string Filename = "mean_reversion";
extern double Lot = 0.1;
extern int MaxPos = 5;
 
void OnTick() {
    // 変数を宣言する。
    bool check;
    int file_handle;
    int i;
    int signal;
    int total;
    int pos;
    int ticket;
    string comment = Filename;

    // シグナルをファイルから確認する。
    file_handle = FileOpen(Filename + ".csv", FILE_READ | FILE_CSV);
    if (file_handle != INVALID_HANDLE) {
        // シグナルには2が加えられているので2を減じて元に戻す。
        signal = FileReadNumber(file_handle) - 2;
        FileClose(file_handle);
    }

    // ポジション総数（このEA以外のものも含む）を確認する。
    total = OrdersTotal();

    // このEAのポジションを確認する。
    pos = 0;
    for (i = 0; i < OrdersTotal(); i++) {
        if (OrderSelect(i, SELECT_BY_POS) == TRUE) {
            if (OrderComment() == comment) {
                if (OrderType() == OP_BUY) {
                    pos = 1;
                }
                else if (OrderType() == OP_SELL) {
                    pos = -1;
                }
            }
        }
    }

    // ポジション総数が最大ポジション数以下である前提で、
    // ポジションが「なし」、かつ、
    if (total <= MaxPos && pos == 0) {
        RefreshRates();
        // シグナルが「買い」なら、新規買い注文を送信する。
        if (signal == 1) {
            ticket = OrderSend(Symbol(), OP_BUY, Lot, Ask, 3, 0, 0, comment);
        }
        // シグナルが「売り」なら、新規売り注文を送信する
        else if (signal == -1) {
            ticket = OrderSend(Symbol(), OP_SELL, Lot, Bid, 3, 0, 0, comment);
        }
    }
    // ポジションが「買い」、かつ、シグナルが「なし」、または「売り」なら、
    // 決済売り注文を送信する。
    else if (pos == 1 && (signal == 0 || signal == -1)) {
        for (i = OrdersTotal() - 1; i >= 0; i--) {
            if(OrderSelect(i, SELECT_BY_POS) == TRUE) {
                if (OrderComment() == comment && OrderType() == OP_BUY) {
                    check = OrderClose(OrderTicket(), OrderLots(), Bid, 3);
                }
            }
        }
    }
    // ポジションが「売り」、かつ、シグナルが「なし」、または「買い」なら、
    // 決済買い注文を送信する。
    else if (pos == -1 && (signal == 0 || signal == 1)) {
        for (i = OrdersTotal() - 1; i >= 0; i--) {
            if(OrderSelect(i, SELECT_BY_POS) == TRUE) {
                if (OrderComment() == comment && OrderType() == OP_SELL) {
                    check = OrderClose(OrderTicket(), OrderLots(), Ask, 3);
                }
            }
        }
    }

    // ポジション総数、ポジション、シグナルを出力する。
    Print("total = ", total);
    Print("pos = ", pos);
    Print("signal = ", signal);
}
