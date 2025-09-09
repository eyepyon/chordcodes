<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chord.codes - AI 音楽生成とリアルタイム演奏アプリ</title>
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
.hero-pattern {
background-image:
radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 2px, transparent 2px),
radial-gradient(circle at 75% 75%, rgba(255,255,255,0.1) 2px, transparent 2px);
background-size: 40px 40px;
}
.card-glow {
box-shadow: 0 0 30px rgba(27, 182, 182, 0.2);
}
.text-gradient {
background: linear-gradient(135deg, #1BB6B6, #1B69B6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
</style>
</head>
<body class="bg-gray-900 text-white">
<header class="gradient-bg relative overflow-hidden">
<div class="hero-pattern absolute inset-0"></div>
<nav class="relative z-10 flex items-center justify-between px-8 py-6">
<div class="flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-white rounded-lg">
<i class="ri-music-2-fill text-primary text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-white">Chord.codes</span>
</div>
<div class="hidden md:flex items-center space-x-8">
<a href="#features" class="text-white hover:text-secondary transition-colors">機能</a>
<a href="#technology" class="text-white hover:text-secondary transition-colors">技術</a>
<a href="#use-cases" class="text-white hover:text-secondary transition-colors">活用例</a>
<a href="#download" class="text-white hover:text-secondary transition-colors">ダウンロード</a>
</div>
<button class="md:hidden w-8 h-8 flex items-center justify-center text-white">
<i class="ri-menu-line text-xl"></i>
</button>
</nav>
<div class="relative z-10 px-8 py-20 md:py-32">
<div class="max-w-6xl mx-auto">
<div class="grid md:grid-cols-2 gap-12 items-center">
<div class="space-y-8">
<h1 class="text-4xl md:text-6xl font-bold leading-tight">
AI が奏でる<br>
<span class="text-secondary">無限の音楽</span>
</h1>
<p class="text-xl text-gray-200 leading-relaxed">
スマートフォンと電子ピアノを USB Type-C で接続し、AI が生成する無限の楽曲をリアルタイムで演奏。ショパン風、ベートーベン風など、あらゆるスタイルの新しい音楽を 24 時間体験できます。
</p>
<div class="flex flex-col sm:flex-row gap-4">
<a href="https://readdy.ai/home/f24f56b8-6ba7-4f71-a8cb-77a197c4023d/0d2d7fdc-19d9-4d8c-b7fc-36fe3335de45" data-readdy="true">
<button class="bg-white text-primary px-8 py-4 !rounded-button font-semibold hover:bg-gray-100 transition-colors whitespace-nowrap">
無料ダウンロード
</button>
</a>
<button id="demo-button" class="border-2 border-white text-white px-8 py-4 !rounded-button font-semibold hover:bg-white hover:text-primary transition-colors whitespace-nowrap">
デモを見る
</button>
<div id="demo-modal" class="fixed inset-0 z-50 hidden">
<div class="absolute inset-0 bg-black bg-opacity-75"></div>
<div class="relative z-10 max-w-4xl mx-auto mt-20 bg-gray-900 rounded-2xl overflow-hidden">
<div class="aspect-w-16 aspect-h-9 bg-black">
<iframe class="w-full h-[480px]" src="about:blank" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<div class="p-8">
<h3 class="text-2xl font-bold mb-4">Chord.codes デモ</h3>
<p class="text-gray-300 mb-6">
AI による音楽生成からリアルタイム演奏まで、Chord.codes の革新的な機能をご覧ください。USB Type-C で簡単接続、直感的な操作で誰でも使いこなせます。
</p>
<div class="flex justify-end">
<button id="close-modal" class="bg-gray-800 text-white px-6 py-2 !rounded-button hover:bg-gray-700 transition-colors">閉じる</button>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="relative">
<img src="https://readdy.ai/api/search-image?query=Modern%20smartphone%20connected%20to%20elegant%20digital%20piano%20via%20USB-C%20cable%2C%20holographic%20musical%20notes%20and%20AI%20circuit%20patterns%20floating%20around%2C%20futuristic%20music%20technology%20setup%20with%20purple%20blue%20gradient%20lighting%2C%20clean%20minimalist%20background&width=600&height=400&seq=hero-main&orientation=landscape" alt="Chord.codes アプリのメイン機能" class="w-full rounded-2xl object-cover object-top">
<div class="absolute -top-4 -right-4 w-20 h-20 bg-secondary rounded-full flex items-center justify-center animate-pulse">
<i class="ri-ai-generate text-white text-2xl"></i>
</div>
</div>
</div>
</div>
</div>
</header>
<section id="features" class="py-20 bg-gray-800">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-16">
<h2 class="text-4xl font-bold mb-6">革新的な機能</h2>
<p class="text-xl text-gray-300 max-w-3xl mx-auto">
独自の AI モデルと軽量 MML 技術により、これまでにない音楽体験を提供します
</p>
</div>
<div class="grid md:grid-cols-3 gap-8">
<div class="bg-gray-900 p-8 rounded-2xl card-glow hover:transform hover:scale-105 transition-all duration-300">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-xl mb-6">
<i class="ri-smartphone-line text-white text-2xl"></i>
</div>
<h3 class="text-2xl font-bold mb-4">USB Type-C 接続</h3>
<p class="text-gray-300 leading-relaxed">
スマートフォンと MIDI 対応電子ピアノを USB Type-C ケーブルで簡単接続。複雑な設定は不要で、すぐに演奏を開始できます。
</p>
</div>
<div class="bg-gray-900 p-8 rounded-2xl card-glow hover:transform hover:scale-105 transition-all duration-300">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-xl mb-6">
<i class="ri-ai-generate text-white text-2xl"></i>
</div>
<h3 class="text-2xl font-bold mb-4">AI 音楽生成</h3>
<p class="text-gray-300 leading-relaxed">
独自開発の AI モデルが、ショパン風、ベートーベン風など様々なスタイルで、聞いたことのない新しい楽曲を無限に生成します。
</p>
</div>
<div class="bg-gray-900 p-8 rounded-2xl card-glow hover:transform hover:scale-105 transition-all duration-300">
<div class="w-16 h-16 flex items-center justify-center bg-gradient-to-br from-primary to-secondary rounded-xl mb-6">
<i class="ri-24-hours-line text-white text-2xl"></i>
</div>
<h3 class="text-2xl font-bold mb-4">24 時間連続演奏</h3>
<p class="text-gray-300 leading-relaxed">
ノンストップで 1 日中楽曲を流し続けることが可能。店舗の BGM や作業用音楽として最適です。
</p>
</div>
</div>
</div>
</section>
<section id="technology" class="py-20 bg-gray-900">
<div class="max-w-6xl mx-auto px-8">
<div class="grid md:grid-cols-2 gap-16 items-center">
<div>
<h2 class="text-4xl font-bold mb-8">
<span class="text-gradient">独自技術</span>による<br>
高品質な音楽生成
</h2>
<div class="space-y-6">
<div class="flex items-start space-x-4">
<div class="w-8 h-8 flex items-center justify-center bg-secondary rounded-full flex-shrink-0 mt-1">
<i class="ri-cpu-line text-white text-sm"></i>
</div>
<div>
<h3 class="text-xl font-semibold mb-2">軽量 MML 拡張技術</h3>
<p class="text-gray-300">
従来の MML を拡張し、より表現豊かで軽量な音楽データ処理を実現
</p>
</div>
</div>
<div class="flex items-start space-x-4">
<div class="w-8 h-8 flex items-center justify-center bg-secondary rounded-full flex-shrink-0 mt-1">
<i class="ri-database-2-line text-white text-sm"></i>
</div>
<div>
<h3 class="text-xl font-semibold mb-2">RAG ライブラリ活用</h3>
<p class="text-gray-300">
膨大な音楽ライブラリから最適な要素を抽出し、リクエストに応じた楽曲を生成
</p>
</div>
</div>
<div class="flex items-start space-x-4">
<div class="w-8 h-8 flex items-center justify-center bg-secondary rounded-full flex-shrink-0 mt-1">
<i class="ri-magic-line text-white text-sm"></i>
</div>
<div>
<h3 class="text-xl font-semibold mb-2">オリジナル AI モデル</h3>
<p class="text-gray-300">
弊社独自開発の AI モデルが、イメージに合った楽曲の生演奏をリアルタイムで提供
</p>
</div>
</div>
</div>
</div>
<div class="relative">
<img src="https://readdy.ai/api/search-image?query=Abstract%20visualization%20of%20AI%20neural%20network%20processing%20musical%20data%2C%20flowing%20digital%20music%20notes%20transforming%20through%20circuit%20patterns%2C%20purple%20blue%20gradient%20background%20with%20geometric%20shapes%20representing%20MML%20data%20processing%20and%20RAG%20library%20connections&width=500&height=600&seq=tech-visual&orientation=portrait" alt="AI 音楽生成技術の可視化" class="w-full rounded-2xl object-cover object-top">
<div class="absolute top-4 right-4 bg-white bg-opacity-20 backdrop-blur-sm rounded-lg p-3">
<div class="flex items-center space-x-2">
<div class="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
<span class="text-sm font-medium">AI 生成中</span>
</div>
</div>
</div>
</div>
</div>
</section>
<section id="use-cases" class="py-20 bg-gray-800">
<div class="max-w-6xl mx-auto px-8">
<div class="text-center mb-16">
<h2 class="text-4xl font-bold mb-6">活用シーン</h2>
<p class="text-xl text-gray-300 max-w-3xl mx-auto">
ストリーミング時代だからこそ価値のある、生演奏のリアルな響きをお客様に
</p>
</div>
<div class="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
<div class="bg-gray-900 p-6 rounded-xl text-center hover:transform hover:scale-105 transition-all duration-300">
<img src="https://readdy.ai/api/search-image?query=Elegant%20restaurant%20interior%20with%20live%20piano%20music%20ambiance%2C%20customers%20enjoying%20dinner%20with%20soft%20lighting%2C%20modern%20sophisticated%20atmosphere%20with%20digital%20piano%20in%20background&width=300&height=200&seq=restaurant&orientation=landscape" alt="レストラン BGM" class="w-full h-32 object-cover object-top rounded-lg mb-4">
<h3 class="text-lg font-semibold mb-2">レストラン BGM</h3>
<p class="text-sm text-gray-400">エレガントな自動演奏でお客様に特別な体験を</p>
</div>
<div class="bg-gray-900 p-6 rounded-xl text-center hover:transform hover:scale-105 transition-all duration-300">
<img src="https://readdy.ai/api/search-image?query=Modern%20office%20workspace%20with%20ambient%20piano%20music%2C%20professionals%20working%20productively%2C%20clean%20minimalist%20design%20with%20digital%20piano%20creating%20peaceful%20work%20environment&width=300&height=200&seq=office&orientation=landscape" alt="オフィス環境音楽" class="w-full h-32 object-cover object-top rounded-lg mb-4">
<h3 class="text-lg font-semibold mb-2">オフィス環境音楽</h3>
<p class="text-sm text-gray-400">集中力を高める作業用 BGM として最適</p>
</div>
<div class="bg-gray-900 p-6 rounded-xl text-center hover:transform hover:scale-105 transition-all duration-300">
<img src="https://readdy.ai/api/search-image?query=Luxury%20hotel%20lobby%20with%20live%20piano%20performance%2C%20guests%20relaxing%20in%20comfortable%20seating%20area%2C%20sophisticated%20interior%20design%20with%20ambient%20lighting%20and%20digital%20piano&width=300&height=200&seq=hotel&orientation=landscape" alt="ホテルロビー" class="w-full h-32 object-cover object-top rounded-lg mb-4">
<h3 class="text-lg font-semibold mb-2">ホテルロビー</h3>
<p class="text-sm text-gray-400">上質な空間演出で特別感を提供</p>
</div>
<div class="bg-gray-900 p-6 rounded-xl text-center hover:transform hover:scale-105 transition-all duration-300">
<img src="https://readdy.ai/api/search-image?query=Cozy%20home%20living%20room%20with%20digital%20piano%2C%20person%20relaxing%20and%20enjoying%20personal%20music%20time%2C%20warm%20comfortable%20atmosphere%20with%20modern%20interior%20design&width=300&height=200&seq=home&orientation=landscape" alt="個人利用" class="w-full h-32 object-cover object-top rounded-lg mb-4">
<h3 class="text-lg font-semibold mb-2">個人利用</h3>
<p class="text-sm text-gray-400">リラックスタイムや音楽鑑賞に</p>
</div>
</div>
</div>
</section>
<section class="py-20 bg-gray-900">
<div class="max-w-4xl mx-auto px-8 text-center">
<h2 class="text-4xl font-bold mb-6">
生演奏の<span class="text-gradient">リアルな響き</span>を<br>
あなたの空間に
</h2>
<p class="text-xl text-gray-300 mb-12 leading-relaxed">
ストリーミングが当たり前の時代だからこそ、生演奏というリアルな奏でをお客様に満喫していただけます。Chord.codes で新しい音楽体験を始めませんか？
</p>
<div class="flex flex-col sm:flex-row gap-6 justify-center items-center mb-12">
<a href="https://readdy.ai/home/f24f56b8-6ba7-4f71-a8cb-77a197c4023d/0d2d7fdc-19d9-4d8c-b7fc-36fe3335de45" data-readdy="true">
<button class="bg-secondary text-white px-10 py-4 !rounded-button font-semibold hover:bg-opacity-90 transition-colors whitespace-nowrap text-lg">
今すぐダウンロード
</button>
</a>
<div class="flex items-center space-x-4 text-gray-400">
<div class="flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-apple-fill text-lg"></i>
</div>
<span>iOS</span>
</div>
<div class="flex items-center space-x-2">
<div class="w-6 h-6 flex items-center justify-center">
<i class="ri-android-fill text-lg"></i>
</div>
<span>Android</span>
</div>
</div>
</div>
<div class="grid md:grid-cols-3 gap-8 text-center">
<div>
<div class="text-3xl font-bold text-secondary mb-2">∞</div>
<div class="text-gray-400">無限の楽曲生成</div>
</div>
<div>
<div class="text-3xl font-bold text-secondary mb-2">24h</div>
<div class="text-gray-400">連続演奏可能</div>
</div>
<div>
<div class="text-3xl font-bold text-secondary mb-2">0ms</div>
<div class="text-gray-400">レイテンシー</div>
</div>
</div>
</div>
</section>
<footer class="bg-gray-900 border-t border-gray-800 py-12">
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
<li><a href="https://readdy.ai/home/f24f56b8-6ba7-4f71-a8cb-77a197c4023d/97764673-021e-44ae-b427-624fc27253e7" data-readdy="true" class="hover:text-white transition-colors">技術仕様</a></li>
<li><a href="#" class="hover:text-white transition-colors">価格</a></li>
<li><a href="#" class="hover:text-white transition-colors">FAQ</a></li>
</ul>
</div>
<div>
<h3 class="font-semibold text-white mb-4">サポート</h3>
<ul class="space-y-2 text-gray-400">
<li><a href="#" class="hover:text-white transition-colors">ヘルプセンター</a></li>
<li><a href="#" class="hover:text-white transition-colors">お問い合わせ</a></li>
<li><a href="https://readdy.ai/home/f24f56b8-6ba7-4f71-a8cb-77a197c4023d/3753fe86-2b6f-4a5a-ae99-212dab989330" data-readdy="true" class="hover:text-white transition-colors">プライバシーポリシー</a></li>
<li><a href="https://readdy.ai/home/f24f56b8-6ba7-4f71-a8cb-77a197c4023d/afe0b60c-c4f0-4c87-9ef2-0c4c10738579" data-readdy="true" class="hover:text-white transition-colors">利用規約</a></li>
</ul>
</div>
</div>
<div class="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
<p>&copy; 2025 Chord.codes. All rights reserved.</p>
</div>
</div>
</footer>
<script id="smooth-scroll">
document.addEventListener('DOMContentLoaded', function() {
const links = document.querySelectorAll('a[href^="#"]');
links.forEach(link => {
link.addEventListener('click', function(e) {
e.preventDefault();
const targetId = this.getAttribute('href');
const targetElement = document.querySelector(targetId);
if (targetElement) {
targetElement.scrollIntoView({
behavior: 'smooth',
block: 'start'
});
}
});
});
});
</script>
<script id="mobile-menu">
document.addEventListener('DOMContentLoaded', function() {
const menuButton = document.querySelector('button[class*="md:hidden"]');
const nav = document.querySelector('nav');
if (menuButton) {
menuButton.addEventListener('click', function() {
const mobileMenu = document.getElementById('mobile-menu');
if (mobileMenu) {
mobileMenu.classList.toggle('hidden');
}
});
}
});
</script>
<script id="modal-handler">
document.addEventListener('DOMContentLoaded', function() {
const demoButton = document.getElementById('demo-button');
const demoModal = document.getElementById('demo-modal');
const closeModal = document.getElementById('close-modal');
if (demoButton && demoModal && closeModal) {
demoButton.addEventListener('click', () => {
demoModal.classList.remove('hidden');
document.body.style.overflow = 'hidden';
const iframe = demoModal.querySelector('iframe');
iframe.src = 'https://www.youtube.com/embed/dQw4w9WgXcQ';
});
closeModal.addEventListener('click', () => {
demoModal.classList.add('hidden');
document.body.style.overflow = '';
const iframe = demoModal.querySelector('iframe');
iframe.src = 'about:blank';
});
demoModal.addEventListener('click', (e) => {
if (e.target === demoModal) {
demoModal.classList.add('hidden');
document.body.style.overflow = '';
const iframe = demoModal.querySelector('iframe');
iframe.src = 'about:blank';
}
});
}
});
</script>
<script id="scroll-effects">
document.addEventListener('DOMContentLoaded', function() {
const cards = document.querySelectorAll('.card-glow');
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
cards.forEach(card => {
card.style.opacity = '0';
card.style.transform = 'translateY(20px)';
card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
observer.observe(card);
});
});
</script>
</body>
</html>