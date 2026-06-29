import sys
from pathlib import Path

print("Python Sürümü:", sys.version)

def run_tests():
    test_results = {}
    
    # 1. app.main import ediliyor mu?
    try:
        from app.main import ALLOWED_MODES, app
        test_results["main_import"] = "Başarılı"
    except Exception as e:
        test_results["main_import"] = f"Başarısız: {e}"
        # Eğer ana import başarısızsa diğer testler anlamsız olabilir
        return test_results

    # 2. allowed_modes içinde geometric_logo var mı?
    if "geometric_logo" in ALLOWED_MODES:
        test_results["allowed_modes_check"] = "Başarılı"
    else:
        test_results["allowed_modes_check"] = "Başarısız"

    # 3. analyzer import ve fonksiyon kontrolü
    try:
        from app.analyzer import analyze_image_from_mem
        # Gerçek bir görselle test etmek gerekir, şimdilik sadece import kontrolü
        test_results["analyzer_import"] = "Başarılı"
    except Exception as e:
        test_results["analyzer_import"] = f"Başarısız: {e}"

    # 5. build_vector_candidates kontrolü
    try:
        from app.vector_engines import build_vector_candidates
        candidates = build_vector_candidates("geometric_logo")
        if len(candidates) >= 4:
            test_results["build_candidates_count"] = "Başarılı"
        else:
            test_results["build_candidates_count"] = f"Başarısız: {len(candidates)} aday bulundu, en az 4 bekleniyordu."
    except Exception as e:
        test_results["build_candidates_count"] = f"Başarısız: {e}"

    # 6 & 8. geometry_cleanup import ve fonksiyon kontrolü
    try:
        from app.geometry_cleanup import cleanup_svg_geometry
        test_results["geometry_cleanup_import"] = "Başarılı"
    except Exception as e:
        test_results["geometry_cleanup_import"] = f"Başarısız: {e}"

    # 9. vectorize_geometric_contours_to_svg import
    try:
        from app.vector_engines import vectorize_geometric_contours_to_svg
        test_results["opencv_vectorizer_import"] = "Başarılı"
    except Exception as e:
        test_results["opencv_vectorizer_import"] = f"Başarısız: {e}"

    # 10, 11, 12. Opsiyonel bağımlılık kontrolleri
    from app.vector_engines import POTRACE_PATH, AUTOTRACE_PATH
    test_results["potrace_check"] = "Bulundu" if POTRACE_PATH else "Bulunamadı (Beklenen davranış)"
    test_results["autotrace_check"] = "Bulundu" if AUTOTRACE_PATH else "Bulunamadı (Beklenen davranış)"

    return test_results

if __name__ == "__main__":
    results = run_tests()
    print("\n--- TEST SONUÇLARI ---")
    for test_name, result in results.items():
        print(f"{test_name:<30}: {result}")
    print("----------------------")