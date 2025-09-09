<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>お問い合わせ - Chord.codes</title>
<script src="https://cdn.tailwindcss.com/3.4.16"></script>
<script>
tailwind.config = {
theme: {
extend: {
colors: {
primary: '#2E1B69',
secondary: '#1BB6B6'
},
borderRadius: {
'none': '0px',
'sm': '4px',
DEFAULT: '8px',
'md': '12px',
'lg': '16px',
'xl': '20px',
'2xl': '24px',
'3xl': '32px',
'full': '9999px',
'button': '8px'
}
}
}
}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/remixicon/4.6.0/remixicon.min.css" rel="stylesheet">
<style>
:where([class^="ri-"])::before {
content: "\f3c2";
}
.gradient-bg {
background: linear-gradient(135deg, #2E1B69 0%, #1B69B6 50%, #1BB6B6 100%);
}
.card-glow {
box-shadow: 0 0 30px rgba(27, 182, 182, 0.1);
}
.text-gradient {
background: linear-gradient(135deg, #1BB6B6, #1B69B6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.contact-card {
transition: all 0.3s ease;
}
.contact-card:hover {
transform: translateY(-4px);
box-shadow: 0 12px 40px rgba(27, 182, 182, 0.15);
}
.form-input {
transition: all 0.3s ease;
}
.form-input:focus {
box-shadow: 0 0 0 3px rgba(27, 182, 182, 0.1);
}
.custom-select {
appearance: none;
background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
background-position: right 0.5rem center;
background-repeat: no-repeat;
background-size: 1.5em 1.5em;
}
.faq-item {
transition: all 0.3s ease;
}
.faq-item:hover {
transform: translateY(-2px);
box-shadow: 0 8px 25px rgba(27, 182, 182, 0.15);
}
.faq-answer {
max-height: 0;
overflow: hidden;
transition: max-height 0.3s ease;
}
.faq-answer.active {
max-height: 300px;
}
</style>
</head>
<body class="bg-gray-50 text-gray-900">
<header class="bg-white shadow-sm border-b border-gray-200">
<nav class="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
<div class="flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-primary rounded-lg">
<i class="ri-music-2-fill text-white text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-primary">Chord.codes</span>
</div>
<div class="flex items-center space-x-6">
<a href="/" data-readdy="true" class="flex items-center space-x-2 text-gray-600 hover:text-primary transition-colors">
<i class="ri-arrow-left-line"></i>
<span>ホームに戻る</span>
</a>
<button class="md:hidden w-8 h-8 flex items-center justify-center text-gray-600">
<i class="ri-menu-line text-xl"></i>
</button>
</div>
</nav>
</header>

<main>
<section class="py-16 bg-white">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-12">
<h1 class="text-4xl md:text-5xl font-bold mb-6">
<span class="text-gradient">お問い合わせ</span>
</h1>
<p class="text-xl text-gray-600 leading-relaxed max-w-3xl mx-auto">
お困りのことがございましたら、お気軽にお問い合わせください。サポートチームが迅速かつ丁寧に対応いたします。
</p>
</div>

<div class="grid md:grid-cols-3 gap-8 mb-16">
<div class="contact-card bg-white rounded-2xl p-8 border border-gray-200 card-glow text-center">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mx-auto mb-6">
<i class="ri-mail-line text-white text-2xl"></i>
</div>
<h3 class="text-xl font-bold mb-4 text-gray-900">メールサポート</h3>
<p class="text-gray-600 mb-4 leading-relaxed">詳細な問い合わせや技術的な質問に最適です。通常 24 時間以内にご返信いたします。</p>
<div class="text-sm text-gray-500 space-y-1">
<p><strong>対応時間:</strong> 24 時間受付</p>
<p><strong>返信時間:</strong> 24 時間以内</p>
</div>
</div>

<div class="contact-card bg-white rounded-2xl p-8 border border-gray-200 card-glow text-center">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-green-500 to-green-600 rounded-2xl mx-auto mb-6">
<i class="ri-chat-3-line text-white text-2xl"></i>
</div>
<h3 class="text-xl font-bold mb-4 text-gray-900">チャットサポート</h3>
<p class="text-gray-600 mb-4 leading-relaxed">リアルタイムでの質問や簡単な問題解決に最適です。即座にサポートを受けられます。</p>
<div class="text-sm text-gray-500 space-y-1">
<p><strong>対応時間:</strong> 平日 9:00-18:00</p>
<p><strong>返信時間:</strong> 即座に対応</p>
</div>
</div>

<div class="contact-card bg-white rounded-2xl p-8 border border-gray-200 card-glow text-center">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl mx-auto mb-6">
<i class="ri-phone-line text-white text-2xl"></i>
</div>
<h3 class="text-xl font-bold mb-4 text-gray-900">電話サポート</h3>
<p class="text-gray-600 mb-4 leading-relaxed">緊急の問題や複雑な技術的サポートが必要な場合にご利用ください。</p>
<div class="text-sm text-gray-500 space-y-1">
<p><strong>対応時間:</strong> 平日 10:00-17:00</p>
<p><strong>電話番号:</strong> 0120-123-456</p>
</div>
</div>
</div>

<div class="grid lg:grid-cols-2 gap-12">
<div class="bg-white rounded-2xl p-8 border border-gray-200 card-glow">
<h2 class="text-2xl font-bold mb-8 text-gray-900">お問い合わせフォーム</h2>
<form class="space-y-6">
<div>
<label class="block text-sm font-semibold text-gray-700 mb-3">お問い合わせカテゴリー</label>
<select class="custom-select w-full px-4 py-3 border border-gray-300 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent form-input text-sm pr-8">
<option>カテゴリーを選択してください</option>
<option>アプリの使い方について</option>
<option>MIDI 接続の問題</option>
<option>音楽生成の不具合</option>
<option>課金・支払いについて</option>
<option>アカウント・ログインの問題</option>
<option>技術的な問題</option>
<option>機能改善の提案</option>
<option>その他</option>
</select>
</div>

<div class="grid md:grid-cols-2 gap-6">
<div>
<label class="block text-sm font-semibold text-gray-700 mb-3">お名前</label>
<input type="text" class="w-full px-4 py-3 border border-gray-300 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent form-input text-sm" placeholder="田中 太郎">
</div>
<div>
<label class="block text-sm font-semibold text-gray-700 mb-3">メールアドレス</label>
<input type="email" class="w-full px-4 py-3 border border-gray-300 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent form-input text-sm" placeholder="example@email.com">
</div>
</div>

<div>
<label class="block text-sm font-semibold text-gray-700 mb-3">件名</label>
<input type="text" class="w-full px-4 py-3 border border-gray-300 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent form-input text-sm" placeholder="お問い合わせの件名を入力してください">
</div>

<div>
<label class="block text-sm font-semibold text-gray-700 mb-3">お問い合わせ内容</label>
<textarea rows="6" class="w-full px-4 py-3 border border-gray-300 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent form-input text-sm resize-none" placeholder="お問い合わせ内容を詳しくご記入ください。使用している機種名やエラーメッセージなどの詳細情報もご記載いただけると、より迅速に対応できます。"></textarea>
</div>

<div class="flex items-start space-x-3">
<div class="flex items-center mt-1">
<input type="checkbox" id="privacy-agree" class="hidden">
<label for="privacy-agree" class="flex items-center cursor-pointer">
<div class="w-5 h-5 border-2 border-gray-300 rounded flex items-center justify-center checkbox-custom">
<i class="ri-check-line text-white text-sm hidden check-icon"></i>
</div>
</label>
</div>
<label for="privacy-agree" class="text-sm text-gray-600 cursor-pointer leading-relaxed">
<a href="#" class="text-primary hover:underline">プライバシーポリシー</a> および <a href="#" class="text-primary hover:underline">利用規約</a> に同意します
</label>
</div>

<button type="submit" class="w-full bg-primary text-white py-4 px-6 !rounded-button font-semibold hover:bg-opacity-90 transition-colors whitespace-nowrap">
送信する
</button>
</form>
</div>

<div>
<div class="bg-white rounded-2xl p-8 border border-gray-200 card-glow mb-8">
<h2 class="text-2xl font-bold mb-6 text-gray-900">サポート営業時間</h2>
<div class="space-y-4">
<div class="flex items-center justify-between py-3 border-b border-gray-100">
<span class="font-medium text-gray-700">メールサポート</span>
<span class="text-gray-600">24 時間受付</span>
</div>
<div class="flex items-center justify-between py-3 border-b border-gray-100">
<span class="font-medium text-gray-700">チャットサポート</span>
<span class="text-gray-600">平日 9:00-18:00</span>
</div>
<div class="flex items-center justify-between py-3 border-b border-gray-100">
<span class="font-medium text-gray-700">電話サポート</span>
<span class="text-gray-600">平日 10:00-17:00</span>
</div>
<div class="flex items-center justify-between py-3">
<span class="font-medium text-gray-700">平均応答時間</span>
<span class="text-gray-600">24 時間以内</span>
</div>
</div>
</div>

<div class="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-8">
<h3 class="text-xl font-bold mb-4 text-gray-900">緊急時のサポート</h3>
<p class="text-gray-600 mb-4 leading-relaxed">
アプリが正常に動作しない、重要なイベントでの使用に支障がある等の緊急事態の場合は、電話サポートをご利用ください。
</p>
<div class="flex items-center space-x-2 text-primary font-semibold">
<div class="w-5 h-5 flex items-center justify-center">
<i class="ri-phone-fill"></i>
</div>
<span>0120-123-456</span>
</div>
</div>
</div>
</div>
</div>
</section>

<section class="py-16 bg-gray-50">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-12">
<h2 class="text-3xl font-bold mb-4 text-gray-900">よくある質問</h2>
<p class="text-gray-600">お問い合わせ前に、こちらもご確認ください</p>
</div>

<div class="mb-8">
<div class="relative max-w-2xl mx-auto">
<input type="text" id="faq-search" placeholder="FAQ を検索..." class="w-full px-6 py-4 pl-14 text-lg border border-gray-200 !rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-white">
<div class="absolute left-4 top-1/2 transform -translate-y-1/2 w-6 h-6 flex items-center justify-center">
<i class="ri-search-line text-gray-400 text-xl"></i>
</div>
</div>
</div>

<div class="space-y-4" id="faq-list">
<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 cursor-pointer">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">MIDI 機器が認識されない場合はどうすればよいですか？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p>USB-C ケーブルの接続確認、電子ピアノの電源確認、アプリの再起動をお試しください。それでも解決しない場合は、機種名とともにサポートまでお問い合わせください。</p>
</div>
</div>
</div>

<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 cursor-pointer">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">音楽生成が遅い場合の対処法は？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p>Wi-Fi 接続の確認、バックグラウンドアプリの終了、音質設定を「標準」に変更してお試しください。ネットワーク環境の改善も効果的です。</p>
</div>
</div>
</div>

<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 cursor-pointer">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">プレミアム版の解約方法を教えてください</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p>iOS: 設定 > Apple ID > サブスクリプション > Chord.codes > サブスクリプションをキャンセル<br>Android: Google Play ストア > メニュー > 定期購入 > Chord.codes > 解約</p>
</div>
</div>
</div>

<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 cursor-pointer">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">対応している電子ピアノの機種は？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p>USB-C ポート搭載の MIDI 対応電子ピアノであれば、ほとんどの機種でご利用いただけます。Yamaha P シリーズ、Roland FP シリーズ、Casio PX シリーズなどが推奨機種です。</p>
</div>
</div>
</div>

<div class="faq-item bg-white rounded-xl p-6 border border-gray-200 cursor-pointer">
<div class="flex items-center justify-between">
<h3 class="text-lg font-semibold text-gray-900 pr-4">アプリがクラッシュする場合の対処法は？</h3>
<div class="w-6 h-6 flex items-center justify-center flex-shrink-0">
<i class="ri-add-line text-gray-400 text-xl faq-icon"></i>
</div>
</div>
<div class="faq-answer mt-4">
<div class="text-gray-600 leading-relaxed">
<p>アプリの完全終了と再起動、デバイスの再起動、アプリの再インストール、OS のアップデート確認、ストレージ容量の確認をお試しください。</p>
</div>
</div>
</div>
</div>

<div class="text-center mt-12">
<p class="text-gray-600 mb-4">他にもご質問がございますか？</p>
<a href="#" class="inline-flex items-center space-x-2 text-primary font-semibold hover:underline">
<span>FAQ ページで詳細を見る</span>
<div class="w-4 h-4 flex items-center justify-center">
<i class="ri-arrow-right-line"></i>
</div>
</a>
</div>
</div>
</section>
</main>

<footer class="bg-gray-900 text-white py-12">
<div class="max-w-6xl mx-auto px-8">
<div class="grid md:grid-cols-4 gap-8">
<div class="md:col-span-2">
<div class="flex items-center space-x-3 mb-6">
<div class="w-10 h-10 flex items-center justify-center bg-white rounded-lg">
<i class="ri-music-2-fill text-primary text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-white">Chord.codes</span>
</div>
<p class="text-gray-400 leading-relaxed mb-6">
AI が生成する無限の音楽をリアルタイムで演奏する革新的なアプリケーション。生演奏の魅力を現代のテクノロジーで再定義します。
</p>
<div class="flex space-x-4">
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-twitter-x-line text-white"></i>
</a>
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-youtube-line text-white"></i>
</a>
<a href="#" class="w-10 h-10 flex items-center justify-center bg-gray-800 rounded-lg hover:bg-secondary transition-colors">
<i class="ri-github-line text-white"></i>
</a>
</div>
</div>
<div>
<h3 class="font-semibold text-white mb-4">製品</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="#" class="hover:text-white transition-colors">機能紹介</a></li>
<li><a href="#" class="hover:text-white transition-colors">技術仕様</a></li>
<li><a href="#" class="hover:text-white transition-colors">価格</a></li>
<li><a href="faq.php" class="hover:text-white transition-colors">FAQ</a></li>
</ul>
</div>
<div>
<h3 class="font-semibold text-white mb-4">サポート</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="support.php" class="hover:text-white transition-colors">お問い合わせ</a></li>
<li><a href="privacy.php" class="hover:text-white transition-colors">プライバシーポリシー</a></li>
<li><a href="terms.php" class="hover:text-white transition-colors">利用規約</a></li>
</ul>
</div>
</div>
<div class="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
<p>&copy; 2025 Chord.codes. All rights reserved.</p>
</div>
</div>
</footer>

<script id="form-functionality">
document.addEventListener('DOMContentLoaded', function() {
const checkbox = document.getElementById('privacy-agree');
const checkboxCustom = document.querySelector('.checkbox-custom');
const checkIcon = document.querySelector('.check-icon');
const submitButton = document.querySelector('button[type="submit"]');

checkbox.addEventListener('change', function() {
if (this.checked) {
checkboxCustom.classList.add('bg-primary', 'border-primary');
checkboxCustom.classList.remove('border-gray-300');
checkIcon.classList.remove('hidden');
submitButton.disabled = false;
submitButton.classList.remove('opacity-50', 'cursor-not-allowed');
} else {
checkboxCustom.classList.remove('bg-primary', 'border-primary');
checkboxCustom.classList.add('border-gray-300');
checkIcon.classList.add('hidden');
submitButton.disabled = true;
submitButton.classList.add('opacity-50', 'cursor-not-allowed');
}
});

submitButton.disabled = true;
submitButton.classList.add('opacity-50', 'cursor-not-allowed');

const form = document.querySelector('form');
form.addEventListener('submit', function(e) {
e.preventDefault();
if (checkbox.checked) {
alert('お問い合わせを送信しました。24時間以内にご返信いたします。');
form.reset();
checkbox.checked = false;
checkboxCustom.classList.remove('bg-primary', 'border-primary');
checkboxCustom.classList.add('border-gray-300');
checkIcon.classList.add('hidden');
submitButton.disabled = true;
submitButton.classList.add('opacity-50', 'cursor-not-allowed');
}
});
});
</script>

<script id="faq-functionality">
document.addEventListener('DOMContentLoaded', function() {
const faqItems = document.querySelectorAll('.faq-item');
const searchInput = document.getElementById('faq-search');

faqItems.forEach(item => {
item.addEventListener('click', function() {
const answer = this.querySelector('.faq-answer');
const icon = this.querySelector('.faq-icon');
const isActive = answer.classList.contains('active');

faqItems.forEach(otherItem => {
const otherAnswer = otherItem.querySelector('.faq-answer');
const otherIcon = otherItem.querySelector('.faq-icon');
otherAnswer.classList.remove('active');
otherIcon.classList.remove('ri-subtract-line');
otherIcon.classList.add('ri-add-line');
});

if (!isActive) {
answer.classList.add('active');
icon.classList.remove('ri-add-line');
icon.classList.add('ri-subtract-line');
}
});
});

searchInput.addEventListener('input', function() {
const searchTerm = this.value.toLowerCase();
faqItems.forEach(item => {
const question = item.querySelector('h3').textContent.toLowerCase();
const answer = item.querySelector('.faq-answer').textContent.toLowerCase();
if (question.includes(searchTerm) || answer.includes(searchTerm)) {
item.style.display = 'block';
} else {
item.style.display = 'none';
}
});
});
});
</script>

<script id="scroll-animations">
document.addEventListener('DOMContentLoaded', function() {
const cards = document.querySelectorAll('.contact-card, .faq-item');
const observer = new IntersectionObserver((entries) => {
entries.forEach(entry => {
if (entry.isIntersecting) {
entry.target.style.opacity = '1';
entry.target.style.transform = 'translateY(0)';
}
});
}, {
threshold: 0.1,
rootMargin: '0px 0px -50px 0px'
});

cards.forEach((card, index) => {
card.style.opacity = '0';
card.style.transform = 'translateY(20px)';
card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
observer.observe(card);
});
});
</script>
</body>
</html>