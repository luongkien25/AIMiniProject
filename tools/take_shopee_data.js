async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toCSVCell(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function downloadCSV(filename, rows) {
  const csv = "review_id,review_text\n" + rows
    .map((reviewText, index) => `${index + 1},${toCSVCell(reviewText)}`)
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
        if (/Ph\u1ea3n H\u1ed3i C\u1ee7a Ng\u01b0\u1eddi B\u00e1n/i.test(next)) break;
        if (/H\u1eefu \u00edch\??/i.test(next)) break;

        if (/^Ph\u00e2n lo\u1ea1i h\u00e0ng:/i.test(next)) continue;
        if (/^[\u2605\u2606\s]+$/.test(next)) continue;
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
    console.log(`Moving to page ${page}...`);

    clickPageNumber(page);
    await sleep(2500);

    const reviews = getReviewsFromVisibleText();

    console.log(`Page ${page}: collected ${reviews.length} reviews`);

    for (const review of reviews) {
      if (!seen.has(review)) {
        seen.add(review);
        all.push(review);
      }
    }
  }

  console.log(`Done. Total unique reviews: ${all.length}`);
  console.log(all);

  downloadCSV("shopee_reviews_all_pages.csv", all);
}

collectShopeeReviewsByPage(1, 20);
