// HTML特殊文字をエスケープする簡単な関数

document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const convertButton = document.getElementById('convertButton');
    const resultArea = document.getElementById('resultArea');
    const errorArea = document.getElementById('errorArea');
    // 2025-04-21 時点でのAPIエンドポイント
    // const apiUrl = 'https://momoapi.sei-ken.dev/convert/';
    const apiUrl = "http://sentaro.local:8080/convert/"; // ローカル開発用


    convertButton.addEventListener('click', convertToBraille);
    // 初期表示のプレースホルダーも p タグに
    resultArea.innerHTML = '<p>ここに変換結果が表示されます。</p>';

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    async function convertToBraille(event) {
        event.preventDefault(); // Prevent the default form submission
        const text = inputText.value.trim();
        resultArea.innerHTML = '<p>変換中...</p>'; // 変換中メッセージをpタグで表示
        errorArea.textContent = '';

        if (!text) {
            resultArea.innerHTML = '<p>ここに変換結果が表示されます。</p>'; // プレースホルダーもpタグに
            errorArea.textContent = 'テキストを入力してください。';
            return;
        }

        try {
            const formData = new FormData(document.getElementById("braille-form"));
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorData = null;
                try {
                    errorData = await response.json();
                } catch (e) {
                    console.error("Failed to parse error response:", e);
                }
                const errorMessage = errorData?.detail
                    ? `エラー: ${errorData.detail} (ステータス: ${response.status})`
                    : `APIリクエストに失敗しました。ステータス: ${response.status}`;
                throw new Error(errorMessage);
            }

            const data = await response.json();

            if (data && data.src !== undefined && data.kana !== undefined && data.braille !== undefined) {
                resultArea.innerHTML = '';
                const srcP = document.createElement('p');
                srcP.innerHTML = `<strong>変換元 (src):</strong><br>${escapeHtml(data.src)}`;
                resultArea.appendChild(srcP);
                const kanaP = document.createElement('p');
                kanaP.innerHTML = `<strong>カタカナ (kana):</strong><br>${escapeHtml(data.kana)}`;
                resultArea.appendChild(kanaP);
                const brailleP = document.createElement('p');
                brailleP.innerHTML = `<strong>点字 (braille):</strong><br><span class="braille-text">${escapeHtml(data.braille)}</span>`;
                // 点字表示用のクラスを付与（CSSでスタイル調整可能）
                resultArea.appendChild(brailleP);

            } else {
                // 想定したキーが存在しない場合のエラー
                console.error("Unexpected response format:", data);
                resultArea.innerHTML = '';
                throw new Error('APIからのレスポンス形式が期待と異なります。(src, kana, brailleが必要です)');
            }
        } catch (error) {
            console.error('変換エラー:', error);
            resultArea.innerHTML = '<p>接続に失敗しました。</p>';
            errorArea.textContent = error.message;
        }
    }
});
