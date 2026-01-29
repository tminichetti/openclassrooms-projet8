"""
Script de test pour l'API de segmentation
"""

import requests
import sys
from pathlib import Path

def test_api(api_url="http://localhost:5000", image_path=None):
    """
    Test l'API de segmentation

    Args:
        api_url: URL de l'API (local ou Heroku)
        image_path: Chemin vers une image de test
    """
    print(f"🧪 Test de l'API: {api_url}")
    print("=" * 60)

    # Test 1: Route de base
    print("\n1️⃣ Test GET /")
    try:
        response = requests.get(f"{api_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ OK")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    # Test 2: Route de santé
    print("\n2️⃣ Test GET /health")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {data}")
        print(f"   Modèle chargé: {data.get('model_loaded')}")
        assert response.status_code == 200
        print("   ✅ OK")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    # Test 3: Prédiction avec image
    if image_path and Path(image_path).exists():
        print(f"\n3️⃣ Test POST /predict avec image: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{api_url}/predict", files=files)

            print(f"   Status: {response.status_code}")
            data = response.json()

            if response.status_code == 200:
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                shape = data.get('shape', {})
                print(f"   Shape mask: {shape.get('height')}x{shape.get('width')}")
                print("   ✅ OK")
            else:
                print(f"   Erreur: {data.get('error')}")
                print(f"   Message: {data.get('message')}")
                print("   ❌ ÉCHEC")

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    else:
        print("\n3️⃣ Test POST /predict - SAUTÉ (pas d'image fournie)")
        print("   ℹ️  Pour tester avec une image:")
        print(f"      python test_api.py {api_url} chemin/vers/image.jpg")

    # Test 4: Prédiction sans image (doit échouer)
    print("\n4️⃣ Test POST /predict sans image (doit échouer)")
    try:
        response = requests.post(f"{api_url}/predict")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 400
        print("   ✅ OK (erreur attendue)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("✅ Tests terminés")

if __name__ == "__main__":
    # Paramètres par défaut
    api_url = "http://localhost:5000"
    image_path = None

    # Parser les arguments
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    if len(sys.argv) > 2:
        image_path = sys.argv[2]

    test_api(api_url, image_path)
