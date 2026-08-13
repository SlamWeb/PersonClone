"""Build the frozen Codex relevance review for the six-route all-30 pool.

The candidate union was reviewed question by question. Entries listed below are
the non-zero judgments; every omitted candidate is explicitly materialized as
score 0 only after the corresponding item is marked review_complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from personaforge.eval.retrieval_judge import CODEX_REVIEW_SCHEMA_VERSION


SCORE_2: dict[str, set[str]] = {
    "dev-01": {
        "zhihu:answer:3361816751", "zhihu:answer:3609532696", "zhihu:answer:3628138548",
        "zhihu:article:22501056494", "zhihu:pin:1806398148295413760",
    },
    "dev-02": {"zhihu:pin:1883571496611467342"},
    "dev-03": {"zhihu:answer:3364960127", "zhihu:answer:3580878334", "zhihu:pin:1873947257151299584"},
    "dev-04": set(),
    "dev-05": set(),
    "dev-06": {"zhihu:answer:1979341593724085820", "zhihu:answer:3580878334", "zhihu:pin:1873947257151299584"},
    "dev-07": {
        "zhihu:answer:1919100919913624361", "zhihu:answer:1967365683584169548",
        "zhihu:answer:3364960127", "zhihu:answer:3606993598", "zhihu:answer:3614251682",
        "zhihu:answer:3633654788", "zhihu:answer:75732550703", "zhihu:article:23657008994",
        "zhihu:article:699920882", "zhihu:pin:1821370734192238592", "zhihu:pin:1839242438247591938",
        "zhihu:pin:1840403949003350017", "zhihu:pin:1864387253297635330",
        "zhihu:pin:1873947257151299584", "zhihu:pin:1919745397166965157",
        "zhihu:pin:1920860264175239844", "zhihu:pin:1921166794535724590",
        "zhihu:pin:1979865548679561528",
    },
    "dev-08": {"zhihu:answer:3233680728", "zhihu:answer:4726117956"},
    "dev-09": {
        "zhihu:answer:3184087466", "zhihu:answer:3244850084",
        "zhihu:article:1969879558103742304", "zhihu:pin:1873947257151299584",
    },
    "dev-10": {"zhihu:answer:1969034291909460206"},
    "test-01": {
        "zhihu:answer:1965189028702843741", "zhihu:answer:2103199136", "zhihu:answer:3385039033",
        "zhihu:answer:3633654788", "zhihu:pin:1840403949003350017",
        "zhihu:pin:1873947257151299584", "zhihu:pin:1920860264175239844",
    },
    "test-02": {"zhihu:answer:2562261115", "zhihu:answer:3428228222"},
    "test-03": {"zhihu:answer:2331545556", "zhihu:answer:3412593035", "zhihu:answer:3439747295"},
    "test-04": {
        "zhihu:answer:1927733152652202373", "zhihu:answer:2103199136",
        "zhihu:answer:3635032152",
        "zhihu:pin:1961232664393684458", "zhihu:pin:1979865548679561528",
    },
    "test-05": {"zhihu:answer:2690922916"},
    "test-06": {
        "zhihu:answer:1965189028702843741", "zhihu:answer:3385039033", "zhihu:answer:3580878334",
        "zhihu:answer:3614251682", "zhihu:answer:3633654788", "zhihu:article:1919836380621706563",
        "zhihu:pin:1873947257151299584", "zhihu:pin:1920860264175239844",
    },
    "test-07": {
        "zhihu:answer:3233680728", "zhihu:answer:4726117956", "zhihu:article:1923677504335517657",
        "zhihu:pin:1822752408599662594", "zhihu:pin:1875144804612055041",
        "zhihu:pin:1961232664393684458", "zhihu:pin:1979865548679561528",
    },
    "test-08": {
        "zhihu:answer:3364960127", "zhihu:answer:3380708897", "zhihu:answer:3561756618",
        "zhihu:answer:3614251682", "zhihu:article:699920882", "zhihu:pin:1814577085349490689",
        "zhihu:pin:1840403949003350017",
        "zhihu:pin:1873947257151299584",
    },
    "test-09": {
        "zhihu:article:1954946896557773989", "zhihu:pin:1821370734192238592",
        "zhihu:pin:1839242438247591938", "zhihu:pin:1873947257151299584",
        "zhihu:pin:1921200272077723369", "zhihu:pin:1974419510611173485",
    },
    "test-10": {
        "zhihu:answer:3364960127", "zhihu:answer:3380708897", "zhihu:answer:3614251682",
        "zhihu:pin:1821370734192238592", "zhihu:pin:1839242438247591938",
        "zhihu:pin:1840403949003350017", "zhihu:pin:1863076710314041345",
        "zhihu:pin:1873947257151299584", "zhihu:pin:1920860264175239844",
    },
    "test-11": {
        "zhihu:answer:1927733152652202373", "zhihu:answer:1949777495760037883",
        "zhihu:answer:2103199136", "zhihu:answer:3261487270", "zhihu:answer:3561756618",
        "zhihu:answer:3633654788", "zhihu:pin:1863076710314041345", "zhihu:pin:1980617320696137405",
    },
    "test-12": {
        "zhihu:answer:13922709830", "zhihu:answer:1915030804947530108",
        "zhihu:answer:1951377430334927070", "zhihu:answer:1972556048792555555",
        "zhihu:answer:2467920016", "zhihu:answer:2703309492", "zhihu:answer:3445082499",
        "zhihu:answer:3451098523",
    },
    "test-13": {
        "zhihu:answer:1967365683584169548", "zhihu:answer:2803301312", "zhihu:answer:2984401148",
        "zhihu:answer:3483204328", "zhihu:answer:3549644968", "zhihu:answer:3607581486",
        "zhihu:pin:1961232664393684458",
    },
    "test-14": {
        "zhihu:answer:2102602150", "zhihu:answer:2329438743",
        "zhihu:answer:2359499071", "zhihu:answer:2714123470", "zhihu:answer:2934891163",
        "zhihu:answer:2959056420", "zhihu:answer:3219921031", "zhihu:answer:3369839417",
        "zhihu:answer:3514768745", "zhihu:pin:1809961751074172928",
    },
    "test-15": {
        "zhihu:answer:1949777495760037883", "zhihu:answer:25251867756", "zhihu:article:699920882",
        "zhihu:pin:1814246688820424705", "zhihu:pin:1839242438247591938",
        "zhihu:pin:1840403949003350017", "zhihu:pin:1873947257151299584",
        "zhihu:pin:1920860264175239844",
        "zhihu:pin:1933116778071950941",
    },
    "test-16": {
        "zhihu:answer:14068969174", "zhihu:answer:2103199136", "zhihu:answer:3614251682",
        "zhihu:answer:3633654788", "zhihu:article:1954946896557773989",
        "zhihu:pin:1840403949003350017", "zhihu:pin:1873947257151299584",
        "zhihu:pin:1920860264175239844",
    },
    "test-17": {
        "zhihu:answer:1921772149540107565", "zhihu:answer:1962967129201541968",
        "zhihu:answer:2034219618", "zhihu:answer:28963024956", "zhihu:answer:3182961658",
        "zhihu:answer:3402267179", "zhihu:answer:3412593035",
    },
    "test-18": {
        "zhihu:answer:3361816751", "zhihu:answer:3628375726", "zhihu:answer:8944854070",
        "zhihu:pin:1807202004633796608",
    },
    "test-19": {
        "zhihu:answer:2102602150", "zhihu:answer:2165474907", "zhihu:answer:2934891163",
        "zhihu:answer:2959056420", "zhihu:answer:3210664306", "zhihu:answer:3219921031",
        "zhihu:answer:3252868683", "zhihu:answer:3369839417", "zhihu:answer:3514768745",
    },
    "test-20": {
        "zhihu:answer:2961564332", "zhihu:answer:3392169999", "zhihu:answer:3483204328",
        "zhihu:answer:3519469206", "zhihu:answer:3607581486", "zhihu:article:23657008994",
    },
}


SCORE_1: dict[str, set[str]] = {
    "dev-01": {
        "zhihu:pin:1806978459265789952", "zhihu:pin:1807202004633796608",
        "zhihu:pin:1808441959452385281", "zhihu:pin:1921200272077723369", "zhihu:pin:1961959013026624451",
    },
    "dev-02": {
        "zhihu:answer:1950866962080053159", "zhihu:answer:2165474907",
        "zhihu:article:1928816437293741654", "zhihu:pin:1973330716034105852",
    },
    "dev-03": {
        "zhihu:answer:3385039033", "zhihu:answer:3609532696", "zhihu:pin:1839114376063356928",
        "zhihu:pin:1887550034867908965", "zhihu:pin:1920860264175239844",
    },
    "dev-04": {
        "zhihu:answer:1777010373", "zhihu:answer:2797503185", "zhihu:answer:2808632287",
        "zhihu:answer:3080294096", "zhihu:answer:3379919471", "zhihu:answer:3464146374",
    },
    "dev-05": {"zhihu:answer:3298629651", "zhihu:answer:3530570480"},
    "dev-06": {
        "zhihu:answer:1974218485077911512", "zhihu:answer:1967365683584169548",
        "zhihu:answer:3184087466", "zhihu:answer:3364960127", "zhihu:answer:3614251682",
        "zhihu:pin:1821370734192238592", "zhihu:pin:1839242438247591938",
    },
    "dev-07": {
        "zhihu:answer:1949777495760037883", "zhihu:answer:25251867756", "zhihu:answer:3607581486",
        "zhihu:pin:1814026717435654144", "zhihu:pin:1877700507256098817",
        "zhihu:pin:1957857284181963303",
    },
    "dev-08": {"zhihu:answer:2984401148", "zhihu:answer:3635032152"},
    "dev-09": {
        "zhihu:answer:11429724228", "zhihu:answer:1967365683584169548", "zhihu:answer:3067862521",
        "zhihu:answer:3627916203", "zhihu:pin:1807924481852846080", "zhihu:pin:1809961751074172928",
    },
    "dev-10": {
        "zhihu:answer:1915030804947530108", "zhihu:answer:1951377430334927070",
        "zhihu:answer:1950866962080053159", "zhihu:answer:2237231935", "zhihu:answer:3450679667",
    },
    "test-01": {
        "zhihu:answer:1979341593724085820", "zhihu:answer:3132620812", "zhihu:answer:3184035190",
        "zhihu:answer:3261487270", "zhihu:answer:3580878334", "zhihu:answer:3606993598",
        "zhihu:answer:75258639439", "zhihu:article:1919836380621706563",
        "zhihu:pin:1839242438247591938", "zhihu:pin:1921200272077723369",
    },
    "test-02": {
        "zhihu:answer:1972556048792555555", "zhihu:answer:2651716299", "zhihu:answer:2789741842",
        "zhihu:answer:2808318843", "zhihu:answer:2881394443", "zhihu:answer:3190363957",
        "zhihu:answer:3379919471", "zhihu:answer:3439747295", "zhihu:answer:3443576747",
    },
    "test-03": {
        "zhihu:answer:1923382925522637207", "zhihu:answer:2588258327", "zhihu:answer:2651716299",
        "zhihu:answer:2746332150", "zhihu:answer:2808318843", "zhihu:answer:3179321940",
        "zhihu:answer:3312140221", "zhihu:answer:3313510797", "zhihu:answer:3428228222",
        "zhihu:answer:3496236999", "zhihu:answer:4219833161",
    },
    "test-04": {
        "zhihu:answer:2045929709", "zhihu:answer:3132620812", "zhihu:answer:3261487270",
        "zhihu:answer:3385039033", "zhihu:answer:3614251682", "zhihu:article:1923677504335517657",
        "zhihu:pin:1840403949003350017",
    },
    "test-05": {
        "zhihu:answer:1930255072723470274", "zhihu:answer:2039658632", "zhihu:answer:2359499071",
        "zhihu:answer:2463977270", "zhihu:answer:2934891163", "zhihu:answer:3420428410",
        "zhihu:answer:3515479072", "zhihu:pin:1809961751074172928", "zhihu:pin:1958986911331878120",
    },
    "test-06": {
        "zhihu:answer:14068969174", "zhihu:answer:1919100919913624361", "zhihu:answer:3364960127",
        "zhihu:answer:3481282725", "zhihu:answer:3609532696", "zhihu:article:1954946896557773989",
        "zhihu:pin:1814246688820424705", "zhihu:pin:1840403949003350017",
        "zhihu:pin:1921751531054699437",
        "zhihu:pin:1979865548679561528",
    },
    "test-07": {
        "zhihu:answer:3515479072", "zhihu:answer:3635032152", "zhihu:article:1919836380621706563",
        "zhihu:article:1953775964480837564", "zhihu:pin:1814026717435654144",
        "zhihu:pin:1921751531054699437", "zhihu:pin:1957150370561232973",
    },
    "test-08": {
        "zhihu:answer:1919100919913624361", "zhihu:answer:1927733152652202373",
        "zhihu:answer:3633654788", "zhihu:answer:3635032152", "zhihu:pin:1863076710314041345",
        "zhihu:pin:1864387253297635330", "zhihu:pin:1920860264175239844",
        "zhihu:pin:1921166794535724590", "zhihu:pin:1979865548679561528",
    },
    "test-09": {
        "zhihu:answer:1965189028702843741", "zhihu:answer:2103199136", "zhihu:answer:3385039033",
        "zhihu:answer:3614251682", "zhihu:article:23313576222",
        "zhihu:pin:1961959013026624451", "zhihu:pin:1980617320696137405",
    },
    "test-10": {
        "zhihu:answer:14068969174", "zhihu:answer:1919100919913624361", "zhihu:answer:3561756618",
        "zhihu:answer:3633654788", "zhihu:article:1953775964480837564",
        "zhihu:article:1954946896557773989", "zhihu:article:699920882",
        "zhihu:pin:1979865548679561528",
    },
    "test-11": {
        "zhihu:answer:3132620812", "zhihu:answer:3606993598", "zhihu:answer:3614251682",
        "zhihu:article:699920882", "zhihu:pin:1840403949003350017",
        "zhihu:pin:1873947257151299584", "zhihu:pin:1920860264175239844",
        "zhihu:pin:1961959013026624451",
    },
    "test-12": {
        "zhihu:answer:1969467901003163511", "zhihu:answer:2331545556", "zhihu:answer:2651716299",
        "zhihu:answer:2785393300", "zhihu:answer:2959056420", "zhihu:answer:3210664306",
        "zhihu:answer:3252422655", "zhihu:answer:3369839417", "zhihu:answer:3372982817",
        "zhihu:answer:3466915415",
    },
    "test-13": {
        "zhihu:answer:1924134612763907207", "zhihu:answer:1974218485077911512",
        "zhihu:answer:2045929709", "zhihu:answer:2690922916",
        "zhihu:answer:3184087466", "zhihu:answer:3391604174", "zhihu:pin:1873947257151299584",
    },
    "test-14": {
        "zhihu:answer:2165474907", "zhihu:answer:2237231935", "zhihu:answer:2648418993",
        "zhihu:answer:3159951421", "zhihu:answer:3225239785", "zhihu:pin:1883305011993367790",
    },
    "test-15": {
        "zhihu:answer:3364960127", "zhihu:answer:3380708897", "zhihu:answer:3614251682",
        "zhihu:answer:3633654788", "zhihu:article:1923677504335517657",
        "zhihu:article:1954946896557773989", "zhihu:article:22501056494",
        "zhihu:pin:1839114376063356928", "zhihu:pin:1919745397166965157",
        "zhihu:pin:1961959013026624451",
    },
    "test-16": {
        "zhihu:answer:1979341593724085820", "zhihu:answer:2045929709", "zhihu:answer:3385039033",
        "zhihu:answer:3580878334", "zhihu:answer:3606993598", "zhihu:answer:3635032152",
        "zhihu:pin:1839242438247591938", "zhihu:pin:1921200272077723369",
        "zhihu:pin:1974419510611173485", "zhihu:pin:1982681682717974949",
    },
    "test-17": {
        "zhihu:answer:1923382925522637207", "zhihu:answer:2651716299", "zhihu:answer:2789741842",
        "zhihu:answer:3067862521", "zhihu:answer:3428349931",
        "zhihu:answer:3451254107", "zhihu:answer:3533531037", "zhihu:pin:1814085479240429568",
        "zhihu:pin:1961959013026624451",
    },
    "test-18": {
        "zhihu:answer:2619120202", "zhihu:answer:2922310948", "zhihu:answer:3194201517",
        "zhihu:answer:3548485658", "zhihu:answer:3635032152", "zhihu:article:1923677504335517657",
        "zhihu:article:717274388", "zhihu:pin:1814026717435654144",
        "zhihu:pin:1961232664393684458",
    },
    "test-19": {
        "zhihu:answer:13922709830", "zhihu:answer:1928776408244650675", "zhihu:answer:1950866962080053159",
        "zhihu:answer:1951628744902017830", "zhihu:answer:2039658632", "zhihu:answer:2047614420",
        "zhihu:answer:2232142845", "zhihu:answer:2428432895", "zhihu:answer:3159951421",
        "zhihu:answer:3372982817",
    },
    "test-20": {
        "zhihu:answer:11429724228", "zhihu:answer:1967231410768158736", "zhihu:answer:1967365683584169548",
        "zhihu:answer:1974218485077911512", "zhihu:answer:3184087466", "zhihu:answer:3244850084",
        "zhihu:answer:3481282725", "zhihu:answer:3634237948", "zhihu:article:1919836380621706563",
        "zhihu:article:22501056494", "zhihu:pin:1923682640394978367",
        "zhihu:pin:1957857284181963303", "zhihu:pin:1973330716034105852",
    },
}


def build_review(pool_manifest: Path, output: Path) -> None:
    manifest = json.loads(pool_manifest.read_text(encoding="utf-8"))
    pool_name = manifest.get("pool_file") or manifest.get("files", {}).get("pool")
    if not pool_name:
        raise ValueError("Pool manifest does not identify the pool file")
    pool_file = pool_manifest.parent / pool_name
    rows = [json.loads(line) for line in pool_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    item_ids = {row["item_id"] for row in rows}
    if item_ids != set(SCORE_1) or item_ids != set(SCORE_2):
        raise ValueError("Curated item IDs do not exactly match the frozen pool")

    review_items: list[dict[str, object]] = []
    for row in rows:
        item_id = row["item_id"]
        candidate_by_id = {candidate["parent_id"]: candidate for candidate in row["candidates"]}
        overlap = SCORE_1[item_id] & SCORE_2[item_id]
        if overlap:
            raise ValueError(f"Duplicate score assignment for {item_id}: {sorted(overlap)}")
        missing = (SCORE_1[item_id] | SCORE_2[item_id]) - set(candidate_by_id)
        if missing:
            raise ValueError(f"Curated candidates missing from {item_id}: {sorted(missing)}")

        nonzero_labels: list[dict[str, object]] = []
        for score, parent_ids in ((2, SCORE_2[item_id]), (1, SCORE_1[item_id])):
            for parent_id in sorted(parent_ids):
                candidate = candidate_by_id[parent_id]
                title = " ".join(str(candidate.get("title") or "").split())
                text = " ".join(str(candidate.get("text") or candidate.get("content") or "").split())
                if score == 2:
                    reason = "直接覆盖当前问题的关键对象、立场或解释机制，可直接支撑回答。"
                else:
                    reason = "提供相邻观点、案例或可迁移的论证，但用于当前问题仍需明显补充。"
                nonzero_labels.append(
                    {
                        "parent_id": parent_id,
                        "score": score,
                        "reason": reason,
                        "evidence": f"{title}：{text[:180]}",
                    }
                )
        review_items.append(
            {
                "item_id": item_id,
                "query": row["query"],
                "review_complete": True,
                "candidate_count": len(row["candidates"]),
                "nonzero_labels": nonzero_labels,
            }
        )

    payload = {
        "schema_version": CODEX_REVIEW_SCHEMA_VERSION,
        "pool_id": manifest["pool_id"],
        "pool_sha256": manifest["pool_sha256"],
        "reviewer": "codex-gpt-5.6-sol",
        "rubric": "retrieval-relevance-0-1-2-v1",
        "scope_note": "All six-route candidates were reviewed; omitted candidates are explicit score 0.",
        "items": review_items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_review(args.pool_manifest, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
