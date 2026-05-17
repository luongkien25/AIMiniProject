from __future__ import annotations

import csv
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd


RANDOM_STATE = 62
BASE_DATA_PATH = Path("data/processed/shopee_reviews_clean_classified_codex.csv")
OUTPUT_PATH = Path("data/processed/generated_neutral_shopee_reviews.csv")
AUGMENTED_OUTPUT_PATH = Path(
    "data/processed/shopee_reviews_clean_classified_codex_neutral_augmented.csv"
)

SOURCE_LABEL = "synthetic_neutral_shopee_realistic_external_informed"

PRODUCTS = [
    "áo",
    "quần",
    "váy",
    "giày",
    "dép",
    "túi",
    "mũ",
    "kẹp tóc",
    "ốp điện thoại",
    "tai nghe",
    "sạc",
    "cáp sạc",
    "đèn ngủ",
    "quạt mini",
    "bình nước",
    "móc treo",
    "thảm",
    "khăn",
    "hộp đựng",
    "son",
    "kem dưỡng",
    "sữa rửa mặt",
    "bút",
    "vở",
    "bánh tráng",
    "khô gà",
    "hạt điều",
    "ly thủy tinh",
    "nồi nhỏ",
    "dao gọt",
    "miếng dán",
    "khẩu trang",
    "vòng tay",
    "nhẫn",
    "dây đeo",
    "kệ để đồ",
]

COLORS = [
    "đen",
    "trắng",
    "xanh",
    "xanh than",
    "hồng",
    "be",
    "nâu",
    "xám",
    "đỏ",
    "tím",
    "vàng nhạt",
    "ghi",
]

MATERIALS = [
    "vải",
    "cotton",
    "thun",
    "nhựa",
    "inox",
    "silicon",
    "da mềm",
    "kim loại",
    "giấy",
    "mút",
    "bạc",
    "len",
    "nỉ",
]

SIZES = [
    "S",
    "M",
    "L",
    "XL",
    "size nhỏ",
    "size vừa",
    "size hơi rộng",
    "size hơi ôm",
    "đúng size",
]

# The phrasing below is newly generated. It is not copied from external datasets.
# It is shaped by common Vietnamese e-commerce review traits: short received-item
# confirmations, "chưa dùng/chưa biết" uncertainty, Shopee-style attribute fields,
# sale/price comments, delivery/packaging mentions, and coin/reward reviews.
TIME_PHRASES = [
    "mới nhận hàng",
    "vừa nhận hôm qua",
    "mới khui ra xem",
    "nhận được chiều nay",
    "hàng mới tới",
    "mình mới kiểm tra sơ",
    "mới lấy từ shipper",
    "vừa mở gói ra xem",
]

NEUTRAL_OBSERVATIONS = [
    "nhìn bên ngoài bình thường",
    "cầm lên thấy ở mức ổn",
    "chưa thấy gì đặc biệt",
    "giống ảnh khoảng tương đối",
    "màu nhìn ngoài hơi khác ánh sáng",
    "form nhìn cơ bản",
    "chất liệu sờ thử tạm được",
    "kích thước nhìn cũng vừa",
    "bao bì nhìn bình thường",
    "không có gì để nhận xét nhiều",
    "hiện tại thấy tạm ổn",
    "mới xem qua nên chưa rõ",
]

PENDING_PHRASES = [
    "để dùng thêm rồi đánh giá sau",
    "dùng một thời gian mới biết",
    "chưa sử dụng nên chưa nói trước được",
    "cần dùng thêm mới biết có bền không",
    "mình sẽ cập nhật nếu có vấn đề",
    "chưa test kỹ nên tạm đánh giá vậy",
    "mới nhận nên chưa đánh giá được nhiều",
    "dùng thử vài hôm rồi tính tiếp",
]

NORMAL_SHIPPING = [
    "giao hàng bình thường",
    "thời gian giao ở mức chấp nhận được",
    "ship không quá nhanh cũng không quá lâu",
    "đơn tới đúng khoảng dự kiến",
    "shipper giao như bình thường",
    "mình nhận hàng sau vài ngày đặt",
]

PACKAGING_PHRASES = [
    "đóng gói bình thường",
    "gói hàng đủ lớp cơ bản",
    "hộp ngoài hơi móp nhẹ nhưng bên trong còn nguyên",
    "bao bì không có gì đặc biệt",
    "túi gói đơn giản",
    "bọc hàng ở mức tạm",
]

SHOP_PHRASES = [
    "chưa trao đổi với shop nhiều",
    "shop phản hồi bình thường",
    "shop xác nhận đơn như mọi lần",
    "mình chưa cần nhắn shop",
    "shop gửi đúng thông tin đơn",
    "tin nhắn tự động là chính",
]

PRICE_PHRASES = [
    "mua lúc sale nên giá tạm ổn",
    "giá ở mức bình thường",
    "giá so với sản phẩm thì tạm chấp nhận",
    "chưa dùng lâu nên chưa biết có đáng tiền không",
    "giá không quá cao",
    "mua thử vì thấy đang giảm giá",
]

SPAM_LIKE_PHRASES = [
    "đánh giá để nhận xu, chưa dùng nên chưa biết",
    "cho đủ ký tự nhận xu thôi, hàng đã nhận",
    "hình ảnh chỉ mang tính nhận xu, sản phẩm chưa dùng",
    "video không liên quan, mình mới nhận hàng",
    "nhận hàng rồi nên lên đánh giá trước",
    "đánh giá sau khi nhận hàng, chưa có trải nghiệm nhiều",
]

SUFFIXES = [
    "",
    "",
    "",
    "",
    " nha",
    " ạ",
    " nhé",
    " thôi",
    " 0:06",
    " 0:10",
    " 0:15",
    " hình ảnh minh họa",
]

STAR_PHRASES = [
    "",
    "",
    "",
    " cho 4 sao trước",
    " tạm để 4 sao",
    " đánh giá tạm vậy",
]

ISSUE_TARGETS = [
    ("no_issue", 430),
    ("product_quality", 190),
    ("product_attribute", 145),
    ("spam_irrelevant", 95),
    ("packaging", 55),
    ("shipping_delivery", 40),
    ("seller_service", 25),
    ("price_value", 20),
]


def choose(items: list[str]) -> str:
    return random.choice(items)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\b(không|ko|kh|k)\b", " khong ", text)
    text = re.sub(r"\b(được|đc|dc)\b", " duoc ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = "".join(
        " " if unicodedata.category(character).startswith("S") else character
        for character in text
    )
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def humanize(text: str) -> str:
    if random.random() < 0.16:
        replacements = [
            ("không", choose(["ko", "k"])),
            ("được", choose(["dc", "đc"])),
            ("mình", choose(["mình", "mk"])),
            ("sản phẩm", choose(["sản phẩm", "sp"])),
            ("bình thường", choose(["bình thường", "bt"])),
            ("với", choose(["với", "vs"])),
        ]
        for old, new in replacements:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)

    if random.random() < 0.12:
        text = text.replace(",", "").replace(".", "")
    if random.random() < 0.08:
        text = text.lower()

    return re.sub(r"\s+", " ", text).strip()


def make_no_issue() -> str:
    product = choose(PRODUCTS)
    patterns = [
        f"{choose(TIME_PHRASES)}, {choose(NEUTRAL_OBSERVATIONS)}, {choose(PENDING_PHRASES)}",
        f"Đã nhận {product}, đủ số lượng, {choose(PENDING_PHRASES)}",
        f"{choose(TIME_PHRASES)} nên chỉ kiểm tra bên ngoài, {choose(NEUTRAL_OBSERVATIONS)}",
        f"Hàng nhận đủ, {choose(NEUTRAL_OBSERVATIONS)}, chưa dùng thực tế",
        f"Mình mua thử {product}, {choose(NEUTRAL_OBSERVATIONS)}, {choose(PENDING_PHRASES)}",
        f"Sản phẩm tới nơi rồi, {choose(NEUTRAL_OBSERVATIONS)}, tạm đánh giá trung bình",
    ]
    return choose(patterns) + choose(STAR_PHRASES)


def make_product_quality() -> str:
    product = choose(PRODUCTS)
    patterns = [
        f"Chất lượng sản phẩm: chưa rõ. {choose(TIME_PHRASES)}, {choose(PENDING_PHRASES)}",
        f"{product.capitalize()} nhìn tạm, chất lượng cần dùng thêm mới biết",
        f"Mới dùng thử {product} một lần, cảm giác bình thường, chưa kết luận được",
        f"Sờ thử thấy chất lượng ở mức trung bình, {choose(PENDING_PHRASES)}",
        "Sản phẩm dùng tạm được, chưa biết lâu dài có ổn không",
        "Vừa bóc ra xem, chất lượng nhìn không có gì đặc biệt, để dùng thêm",
    ]
    return choose(patterns)


def make_product_attribute() -> str:
    product = choose(PRODUCTS)
    color = choose(COLORS)
    material = choose(MATERIALS)
    size = choose(SIZES)
    description = choose(
        [
            "đúng tương đối",
            "khá giống ảnh",
            "chưa chắc lắm",
            "nhìn gần giống mô tả",
            "cơ bản giống hình",
        ]
    )
    patterns = [
        f"Màu sắc: {color} Chất liệu: {material} Đúng với mô tả: {description}. Mới nhận nên chưa dùng",
        f"{product.capitalize()} màu {color}, {size}, nhìn ngoài bình thường, chưa thử lâu",
        "Màu hơi khác ảnh một chút do ánh sáng, còn lại mình chưa dùng nên chưa biết",
        f"Kích thước {size}, chất liệu {material}, cảm nhận ban đầu bình thường",
        f"Đúng mẫu đặt, màu {color}, mình mới kiểm tra sơ nên đánh giá tạm",
        "Form và màu nhìn ở mức chấp nhận được, cần mặc/dùng thêm mới biết",
    ]
    return choose(patterns)


def make_spam_irrelevant() -> str:
    patterns = SPAM_LIKE_PHRASES + [
        f"{choose(['aaaaaaaa', 'abcxyz', '........'])} mình ghi nhận đã nhận hàng",
        "Chưa dùng, đánh giá trước cho shop và nhận xu",
        "Không biết viết gì, hàng mới nhận nên để vậy trước",
        "Ảnh không liên quan lắm, mình chỉ mới kiểm tra đơn hàng",
    ]
    return choose(patterns)


def make_packaging() -> str:
    patterns = [
        f"Bao bì/Mẫu mã: {choose(PACKAGING_PHRASES)}. Mình chưa dùng sản phẩm bên trong",
        f"{choose(PACKAGING_PHRASES).capitalize()}, nhận hàng bình thường",
        "Đóng gói ở mức cơ bản, hàng bên trong chưa kiểm tra kỹ",
        "Hộp nhận được hơi móp nhẹ, sản phẩm bên trong nhìn vẫn bình thường",
        "Gói hàng đơn giản, đủ sản phẩm, chưa sử dụng nên chưa đánh giá thêm",
    ]
    return choose(patterns)


def make_shipping_delivery() -> str:
    patterns = [
        f"{choose(NORMAL_SHIPPING).capitalize()}, hàng nhận đủ, chưa dùng",
        "Thời gian giao hàng bình thường, mình mới nhận nên chưa đánh giá sản phẩm",
        "Đơn tới nơi, ship ở mức ổn, sản phẩm chưa test",
        "Giao hàng hoàn tất, không có gì đặc biệt, để dùng thêm rồi nhận xét sau",
        "Mình nhận hàng sau vài ngày, trạng thái đơn bình thường",
    ]
    return choose(patterns)


def make_seller_service() -> str:
    patterns = [
        f"{choose(SHOP_PHRASES).capitalize()}, hàng mới nhận nên chưa đánh giá thêm",
        "Shop xử lý đơn bình thường, mình chưa dùng sản phẩm",
        "Chưa hỏi shop nhiều, đơn giao như thông tin trên app",
        "Shop nhắn xác nhận đơn, còn sản phẩm mình chưa test kỹ",
        "Tư vấn chưa nhiều nên mình chưa nhận xét được, hàng đã nhận",
    ]
    return choose(patterns)


def make_price_value() -> str:
    patterns = [
        f"{choose(PRICE_PHRASES).capitalize()}, sản phẩm mới nhận nên chưa biết lâu dài",
        "Giá tạm được, chất lượng cần dùng thêm mới đánh giá rõ",
        "Mua thử theo deal, nhận hàng rồi nhưng chưa sử dụng",
        "Với giá này thì mình cần dùng thêm mới biết có hợp không",
        "Mức giá bình thường, hàng nhận đủ, chưa có nhận xét thêm",
    ]
    return choose(patterns)


MAKERS = {
    "no_issue": make_no_issue,
    "product_quality": make_product_quality,
    "product_attribute": make_product_attribute,
    "spam_irrelevant": make_spam_irrelevant,
    "packaging": make_packaging,
    "shipping_delivery": make_shipping_delivery,
    "seller_service": make_seller_service,
    "price_value": make_price_value,
}


def load_existing_reviews() -> tuple[pd.DataFrame, set[str]]:
    base_df = pd.read_csv(BASE_DATA_PATH)
    existing_reviews = set(
        base_df["review_text"].dropna().astype(str).str.strip().str.lower()
    )
    return base_df, existing_reviews


def generate_rows(existing_reviews: set[str]) -> pd.DataFrame:
    rows = []
    seen = set(existing_reviews)
    seen_cleaned = set()

    for issue, target_count in ISSUE_TARGETS:
        current_count = 0
        attempts = 0
        while current_count < target_count:
            attempts += 1
            if attempts > target_count * 300:
                raise RuntimeError(f"Could not generate enough unique rows for {issue}")

            review_text = humanize(MAKERS[issue]() + choose(SUFFIXES))
            review_key = review_text.strip().lower()
            cleaned_review = clean_text(review_text)
            if (
                review_key in seen
                or cleaned_review in seen_cleaned
                or len(review_text.split()) < 5
            ):
                continue

            seen.add(review_key)
            seen_cleaned.add(cleaned_review)
            rows.append(
                {
                    "page": 0,
                    "review_text": review_text,
                    "cleaned_review": cleaned_review,
                    "sentiment": "neutral",
                    "issue": issue,
                    "source": SOURCE_LABEL,
                }
            )
            current_count += 1

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "synthetic_id", [f"neutral_gen_{i:04d}" for i in range(1, len(df) + 1)])
    return df


def save_outputs(base_df: pd.DataFrame, generated_df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    augmented_base = base_df.copy()
    if "source" not in augmented_base.columns:
        augmented_base["source"] = "existing_codex_dataset"

    augmented_df = pd.concat(
        [
            augmented_base,
            generated_df[
                ["page", "review_text", "cleaned_review", "sentiment", "issue", "source"]
            ],
        ],
        ignore_index=True,
    )
    augmented_df.to_csv(
        AUGMENTED_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    return augmented_df


def main() -> None:
    random.seed(RANDOM_STATE)
    base_df, existing_reviews = load_existing_reviews()
    generated_df = generate_rows(existing_reviews)
    augmented_df = save_outputs(base_df, generated_df)

    print("Generated rows:", len(generated_df))
    print("Output:", OUTPUT_PATH)
    print(generated_df["sentiment"].value_counts().to_string())
    print(generated_df["issue"].value_counts().to_string())
    print()
    print("Sample:")
    print(
        generated_df[["review_text", "cleaned_review", "sentiment", "issue"]]
        .head(20)
        .to_string(index=False)
    )
    print()
    print("Augmented output:", AUGMENTED_OUTPUT_PATH)
    print(augmented_df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
