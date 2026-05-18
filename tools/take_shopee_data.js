async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function downloadCSV(filename, rows) {
  const csv = "page,review_text\n" + rows
    .map(r => `${r.page},"${String(r.review_text).replaceAll('"', '""')}"`)
    .join("\n");

  const blob = new Blob(["\uFEFF" + csv], {
    type: "text/csv;charset=utf-8;"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function getReviewsFromVisibleText() {
  const lines = document.body.innerText
    .split("\n")
    .map(x => x.trim())
    .filter(Boolean);

  const reviews = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(line)) {
      const block = [];

      for (let j = i + 1; j < lines.length; j++) {
        const next = lines[j];

        if (/\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/.test(next)) break;
        if (/Phản Hồi Của Người Bán/i.test(next)) break;
        if (/Hữu ích\??/i.test(next)) break;

        if (/^Phân loại hàng:/i.test(next)) continue;
        if (/^[★☆\s]+$/.test(next)) continue;
        if (/^\.\.\.$/.test(next)) continue;

        block.push(next);
      }

      const review = block.join(" ").replace(/\s+/g, " ").trim();

      if (review.length >= 3) {
        reviews.push(review);
      }
    }
  }

  return [...new Set(reviews)];
}

function clickPageNumber(pageNumber) {
  const buttons = [...document.querySelectorAll("button, a")];

  const target = buttons.find(btn => {
    return (btn.innerText || "").trim() === String(pageNumber);
  });

  if (!target) return false;

  target.scrollIntoView({ block: "center" });
  target.click();
  return true;
}

async function collectShopeeReviewsByPage(startPage = 1, endPage = 10) {
  const all = [];
  const seen = new Set();

  for (let page = startPage; page <= endPage; page++) {
    console.log(`Đang chuyển tới trang ${page}...`);
    
    clickPageNumber(page);
    await sleep(2500);

    const reviews = getReviewsFromVisibleText();

    console.log(`Trang ${page}: lấy được ${reviews.length} bình luận`);

    for (const review of reviews) {
      if (!seen.has(review)) {
        seen.add(review);
        all.push({
          page,
          review_text: review
        });
      }
    }
  }

  console.log(`Hoàn tất. Tổng số bình luận không trùng: ${all.length}`);
  console.log(all);

  downloadCSV("shopee_reviews_all_pages.csv", all);
}

collectShopeeReviewsByPage(1, 20);
