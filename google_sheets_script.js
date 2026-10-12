/**
 * GOOGLE APPS SCRIPT CHO HỆ THỐNG DSS TRAVEL (LƯU TRỮ PHẢN HỒI NGƯỜI DÙNG)
 * 
 * Hướng dẫn:
 * 1. Mở Google Sheet mới -> Tiện ích mở rộng (Extensions) -> Apps Script
 * 2. Dán toàn bộ mã này vào file Code.gs
 * 3. Bấm Deploy (Triển khai) -> New deployment (Tùy chọn triển khai mới)
 * 4. Chọn loại: Web app (Ứng dụng web)
 *    - Execute as: Me (Tôi)
 *    - Who has access: Anyone (Bất kỳ ai)
 * 5. Bấm Deploy và copy đường link URL Web app (đuôi /exec)
 * 6. Dán vào Streamlit Cloud Secrets: FEEDBACK_WEBHOOK_URL = "url_cua_ban"
 */

function doGet(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var revSheet = ss.getSheetByName("Reviews");
  var viewSheet = ss.getSheetByName("Views");
  
  var explicit_reviews = [];
  if (revSheet && revSheet.getLastRow() > 1) {
    var revData = revSheet.getRange(2, 1, revSheet.getLastRow() - 1, 7).getValues();
    for (var i = revData.length - 1; i >= 0; i--) { // Đánh giá mới nhất xếp trước
      var row = revData[i];
      var payload = {};
      try {
        if (row[6]) payload = JSON.parse(row[6]);
      } catch (err) {}
      
      explicit_reviews.push({
        timestamp: row[0],
        persona: row[1],
        stars: Number(row[2]) || 0,
        fit_items: row[3] ? String(row[3]).split(", ").filter(Boolean) : [],
        unfit_items: row[4] ? String(row[4]).split(", ").filter(Boolean) : [],
        comment: row[5] || "",
        items: payload.items || [],
        context: payload.context || {}
      });
    }
  }
  
  var implicit_views = {};
  if (viewSheet && viewSheet.getLastRow() > 1) {
    var vData = viewSheet.getRange(2, 1, viewSheet.getLastRow() - 1, 4).getValues();
    for (var j = 0; j < vData.length; j++) {
      var vRow = vData[j];
      var dest_id = String(vRow[0]);
      if (dest_id) {
        implicit_views[dest_id] = {
          name: vRow[1],
          region: vRow[2],
          views: Number(vRow[3]) || 0
        };
      }
    }
  }
  
  var result = {
    explicit_reviews: explicit_reviews,
    implicit_views: implicit_views
  };
  
  return ContentService.createTextOutput(JSON.stringify(result))
    .setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    
    // Đảm bảo có sheet Reviews và Views
    var revSheet = ss.getSheetByName("Reviews");
    if (!revSheet) {
      revSheet = ss.insertSheet("Reviews");
      revSheet.appendRow(["Thời gian", "Ngữ cảnh", "Số sao", "Phù hợp", "Chưa phù hợp", "Bình luận", "Payload"]);
    }
    var viewSheet = ss.getSheetByName("Views");
    if (!viewSheet) {
      viewSheet = ss.insertSheet("Views");
      viewSheet.appendRow(["Mã điểm đến", "Tên điểm đến", "Miền", "Lượt xem"]);
    }
    
    // 1. Thêm đánh giá mới
    if (data.action === "save_explicit") {
      var r = data.review;
      var payloadStr = JSON.stringify({ items: r.items || [], context: r.context || {} });
      revSheet.appendRow([
        r.timestamp || new Date().toISOString(),
        r.persona || "",
        r.stars || 0,
        (r.fit_items || []).join(", "),
        (r.unfit_items || []).join(", "),
        r.comment || "",
        payloadStr
      ]);
      return ContentService.createTextOutput(JSON.stringify({ status: "success" }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    // 2. Ghi nhận lượt xem (Implicit)
    if (data.action === "log_view") {
      var v = data.view;
      var found = false;
      if (viewSheet.getLastRow() > 1) {
        var vData = viewSheet.getRange(2, 1, viewSheet.getLastRow() - 1, 4).getValues();
        for (var i = 0; i < vData.length; i++) {
          if (String(vData[i][0]) === String(v.dest_id)) {
            var currViews = Number(vData[i][3]) || 0;
            viewSheet.getRange(i + 2, 4).setValue(currViews + 1);
            found = true;
            break;
          }
        }
      }
      if (!found) {
        viewSheet.appendRow([v.dest_id, v.name, v.region, 1]);
      }
      return ContentService.createTextOutput(JSON.stringify({ status: "success" }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    // 3. Đồng bộ toàn bộ dữ liệu từ file local lên Sheet
    if (data.action === "sync_all") {
      revSheet.clearContents();
      revSheet.appendRow(["Thời gian", "Ngữ cảnh", "Số sao", "Phù hợp", "Chưa phù hợp", "Bình luận", "Payload"]);
      var reviews = data.explicit_reviews || [];
      for (var k = 0; k < reviews.length; k++) {
        var rev = reviews[k];
        var pStr = JSON.stringify({ items: rev.items || [], context: rev.context || {} });
        revSheet.appendRow([
          rev.timestamp || "",
          rev.persona || "",
          rev.stars || 0,
          (rev.fit_items || []).join(", "),
          (rev.unfit_items || []).join(", "),
          rev.comment || "",
          pStr
        ]);
      }
      
      viewSheet.clearContents();
      viewSheet.appendRow(["Mã điểm đến", "Tên điểm đến", "Miền", "Lượt xem"]);
      var viewsMap = data.implicit_views || {};
      for (var dId in viewsMap) {
        var item = viewsMap[dId];
        viewSheet.appendRow([dId, item.name || "", item.region || "", item.views || 0]);
      }
      return ContentService.createTextOutput(JSON.stringify({ status: "success" }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    // 4. Xóa sạch dữ liệu (Reset)
    if (data.action === "reset") {
      revSheet.clearContents();
      revSheet.appendRow(["Thời gian", "Ngữ cảnh", "Số sao", "Phù hợp", "Chưa phù hợp", "Bình luận", "Payload"]);
      viewSheet.clearContents();
      viewSheet.appendRow(["Mã điểm đến", "Tên điểm đến", "Miền", "Lượt xem"]);
      return ContentService.createTextOutput(JSON.stringify({ status: "success" }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    return ContentService.createTextOutput(JSON.stringify({ status: "unknown action" }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: err.toString() }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
