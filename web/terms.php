<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>利用規約 - Chord.codes</title>
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
.text-gradient {
background: linear-gradient(135deg, #1BB6B6, #1B69B6);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.toc-item {
transition: all 0.2s ease;
}
.toc-item:hover {
background-color: rgba(27, 182, 182, 0.1);
}
.section-content {
scroll-margin-top: 2rem;
}
</style>
</head>
<body class="bg-gray-50 text-gray-900">
<header class="gradient-bg relative overflow-hidden">
<div class="hero-pattern absolute inset-0"></div>
<nav class="relative z-10 flex items-center justify-between px-8 py-6">
<div class="flex items-center space-x-3">
<div class="w-10 h-10 flex items-center justify-center bg-white rounded-lg">
<i class="ri-music-2-fill text-primary text-xl"></i>
</div>
<span class="font-['Pacifico'] text-2xl text-white">Chord.codes</span>
</div>
<div class="flex items-center space-x-6">
<a href="/" data-readdy="true" class="flex items-center space-x-2 text-white hover:text-secondary transition-colors">
<i class="ri-arrow-left-line"></i>
<span>ホームに戻る</span>
</a>
</div>
</nav>
<div class="relative z-10 px-8 py-16">
<div class="max-w-4xl mx-auto text-center">
<h1 class="text-4xl md:text-5xl font-bold text-white mb-4">利用規約</h1>
<p class="text-xl text-gray-200">Chord.codes サービス利用に関する重要事項</p>
</div>
</div>
</header>

<main class="max-w-4xl mx-auto px-8 py-12">
<div class="bg-white rounded-2xl shadow-lg overflow-hidden">
<div class="p-8 border-b border-gray-200">
<div class="flex justify-between items-center">
<h2 class="text-2xl font-bold text-gray-900">利用規約</h2>
<div class="text-sm text-gray-500">最終更新日：2025年9月9日</div>
</div>
</div>

<div class="grid md:grid-cols-4 gap-8 p-8">
<div class="md:col-span-1">
<div class="bg-gray-50 rounded-xl p-6 sticky top-8">
<h3 class="font-semibold text-gray-900 mb-4">目次</h3>
<nav class="space-y-2">
<a href="#section-1" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">1. 総則</a>
<a href="#section-2" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">2. 会社情報</a>
<a href="#section-3" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">3. サービス利用条件</a>
<a href="#section-4" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">4. 禁止事項</a>
<a href="#section-5" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">5. 免責事項</a>
<a href="#section-6" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">6. 個人情報の取り扱い</a>
<a href="#section-7" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">7. 知的財産権</a>
<a href="#section-8" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">8. 契約の変更・終了</a>
<a href="#section-9" class="toc-item block px-3 py-2 text-sm text-gray-600 hover:text-secondary rounded-lg">9. その他</a>
</nav>
<button id="back-to-top" class="mt-6 w-full bg-secondary text-white px-4 py-2 !rounded-button text-sm hover:bg-opacity-90 transition-colors whitespace-nowrap">
トップへ戻る
</button>
</div>
</div>

<div class="md:col-span-3 space-y-12">
<section id="section-1" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">1. 総則</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<p class="text-gray-700 leading-relaxed mb-4">
本利用規約（以下「本規約」といいます）は、フロッグカンパニー株式会社（以下「当社」といいます）が提供する音楽生成・演奏アプリケーション「Chord.codes」（以下「本サービス」といいます）の利用条件を定めるものです。
</p>
<p class="text-gray-700 leading-relaxed mb-4">
ユーザーの皆様（以下「ユーザー」といいます）には、本サービスをご利用いただく前に、本規約の内容をご確認いただき、これに同意いただく必要があります。
</p>
<p class="text-gray-700 leading-relaxed">
本サービスにユーザー登録された時点、または本サービスを利用された時点で、ユーザーは本規約に同意されたものとみなします。
</p>
</div>
</section>

<section id="section-2" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">2. 会社情報</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="bg-gray-50 rounded-xl p-6">
<div class="grid md:grid-cols-2 gap-6">
<div>
<h4 class="font-semibold text-gray-900 mb-3">会社概要</h4>
<ul class="space-y-2 text-gray-700">
<li><strong>会社名：</strong>フロッグカンパニー株式会社</li>
<li><strong>代表者：</strong>代表取締役CTO 會田 昌史</li>
</ul>
</div>
<div>
<h4 class="font-semibold text-gray-900 mb-3">連絡先</h4>
<ul class="space-y-2 text-gray-700">
<li><strong>所在地：</strong>〒106-0044<br>東京都港区東麻布1丁目5-2<br>ザイマックス東麻布ビル8F</li>
<li><strong>電話：</strong>050-3196-9600</li>
<li><strong>メール：</strong>support@chord.codes</li>
</ul>
</div>
</div>
</div>
</section>

<section id="section-3" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">3. サービスの利用条件</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">3.1 利用資格</h4>
<ul class="list-disc list-inside space-y-2 text-gray-700 mb-6">
<li>本サービスは、13歳以上の方にご利用いただけます</li>
<li>未成年者の場合は、保護者の同意が必要です</li>
<li>法人でのご利用の場合は、適切な権限を有する方が登録してください</li>
</ul>

<h4 class="text-lg font-semibold text-gray-900 mb-3">3.2 アカウント管理</h4>
<ul class="list-disc list-inside space-y-2 text-gray-700 mb-6">
<li>ユーザーは正確な情報を提供し、常に最新の状態に保つ責任があります</li>
<li>パスワードの管理はユーザーの責任で行ってください</li>
<li>アカウントの不正利用を発見した場合は、直ちに当社にご連絡ください</li>
</ul>

<h4 class="text-lg font-semibold text-gray-900 mb-3">3.3 技術要件</h4>
<ul class="list-disc list-inside space-y-2 text-gray-700">
<li>iOS 14.0 以降、または Android 8.0 以降の端末</li>
<li>USB Type-C 対応の MIDI 電子ピアノ</li>
<li>安定したインターネット接続環境</li>
</ul>
</div>
</section>

<section id="section-4" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">4. 禁止事項</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<p class="text-gray-700 leading-relaxed mb-4">
ユーザーは、本サービスの利用にあたり、以下の行為を行ってはなりません：
</p>
<div class="bg-red-50 border border-red-200 rounded-xl p-6">
<ul class="list-disc list-inside space-y-2 text-gray-700">
<li>法令、規則、条例等に違反する行為</li>
<li>当社、他のユーザー、または第三者の知的財産権を侵害する行為</li>
<li>本サービスの運営を妨害する行為</li>
<li>不正アクセス、ハッキング、クラッキング等の行為</li>
<li>生成された楽曲を商用目的で無断使用する行為</li>
<li>他のユーザーになりすます行為</li>
<li>虚偽の情報を登録または送信する行為</li>
<li>本サービスを通じて得た情報を第三者に提供する行為</li>
<li>その他、当社が不適切と判断する行為</li>
</ul>
</div>
</div>
</section>

<section id="section-5" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">5. 免責事項</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">5.1 サービスの提供</h4>
<p class="text-gray-700 leading-relaxed mb-4">
当社は、本サービスの品質、正確性、安全性、有用性について、明示または黙示を問わず、いかなる保証も行いません。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">5.2 損害の免責</h4>
<p class="text-gray-700 leading-relaxed mb-4">
当社は、本サービスの利用に関連してユーザーに生じた損害について、当社に故意または重過失がある場合を除き、一切の責任を負いません。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">5.3 第三者との紛争</h4>
<p class="text-gray-700 leading-relaxed">
ユーザーと第三者との間で紛争が生じた場合、ユーザーの責任と費用負担において解決するものとし、当社は一切の責任を負いません。
</p>
</div>
</section>

<section id="section-6" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">6. 個人情報の取り扱い</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">6.1 収集する情報</h4>
<ul class="list-disc list-inside space-y-2 text-gray-700 mb-6">
<li>アカウント登録時に提供いただく情報（メールアドレス、氏名等）</li>
<li>本サービスの利用履歴、操作ログ</li>
<li>デバイス情報、IP アドレス</li>
<li>音楽生成の設定情報、演奏データ</li>
</ul>

<h4 class="text-lg font-semibold text-gray-900 mb-3">6.2 利用目的</h4>
<ul class="list-disc list-inside space-y-2 text-gray-700 mb-6">
<li>本サービスの提供、運営、改善</li>
<li>ユーザーサポート、お問い合わせ対応</li>
<li>新機能、キャンペーン等のご案内</li>
<li>利用状況の分析、統計データの作成</li>
</ul>

<h4 class="text-lg font-semibold text-gray-900 mb-3">6.3 第三者提供</h4>
<p class="text-gray-700 leading-relaxed">
当社は、法令に基づく場合を除き、ユーザーの同意なく個人情報を第三者に提供することはありません。
</p>
</div>
</section>

<section id="section-7" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">7. 知的財産権</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">7.1 当社の知的財産権</h4>
<p class="text-gray-700 leading-relaxed mb-4">
本サービスに関する著作権、商標権、特許権その他の知的財産権は、当社または当社にライセンスを許諾している第三者に帰属します。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">7.2 生成楽曲の取り扱い</h4>
<div class="bg-blue-50 border border-blue-200 rounded-xl p-6">
<ul class="list-disc list-inside space-y-2 text-gray-700">
<li>AI により生成された楽曲の著作権は当社に帰属します</li>
<li>個人利用の範囲内での演奏、録音は許可されます</li>
<li>商用利用については別途ライセンス契約が必要です</li>
<li>楽曲の改変、二次創作は禁止されています</li>
</ul>
</div>

<h4 class="text-lg font-semibold text-gray-900 mb-3 mt-6">7.3 ユーザーコンテンツ</h4>
<p class="text-gray-700 leading-relaxed">
ユーザーが本サービスに投稿したコンテンツについて、当社は本サービスの運営に必要な範囲で利用する権利を有します。
</p>
</div>
</section>

<section id="section-8" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">8. 契約の変更・終了</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">8.1 規約の変更</h4>
<p class="text-gray-700 leading-relaxed mb-4">
当社は、必要に応じて本規約を変更することがあります。重要な変更については、本サービス内またはメールにて事前に通知いたします。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">8.2 サービスの終了</h4>
<p class="text-gray-700 leading-relaxed mb-4">
当社は、30日前の事前通知により、本サービスの全部または一部を終了することができます。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">8.3 アカウントの停止・削除</h4>
<p class="text-gray-700 leading-relaxed">
当社は、ユーザーが本規約に違反した場合、事前の通知なくアカウントを停止または削除することができます。
</p>
</div>
</section>

<section id="section-9" class="section-content">
<div class="flex justify-between items-center mb-6">
<h3 class="text-2xl font-bold text-gray-900">9. その他</h3>
<button class="scroll-to-top text-secondary hover:text-primary transition-colors">
<i class="ri-arrow-up-line text-lg"></i>
</button>
</div>
<div class="prose prose-gray max-w-none">
<h4 class="text-lg font-semibold text-gray-900 mb-3">9.1 準拠法・管轄裁判所</h4>
<p class="text-gray-700 leading-relaxed mb-4">
本規約は日本法に準拠し、本規約に関する一切の紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とします。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">9.2 分離可能性</h4>
<p class="text-gray-700 leading-relaxed mb-4">
本規約の一部が無効または執行不能と判断された場合でも、残りの部分の有効性には影響しません。
</p>

<h4 class="text-lg font-semibold text-gray-900 mb-3">9.3 お問い合わせ</h4>
<div class="bg-gray-50 rounded-xl p-6">
<p class="text-gray-700 leading-relaxed mb-3">
本規約に関するご質問やお問い合わせは、以下までご連絡ください：
</p>
<ul class="space-y-1 text-gray-700">
<li><strong>メール：</strong>legal@chord.codes</li>
<li><strong>電話：</strong>03-6804-5678（平日 10:00-18:00）</li>
<li><strong>お問い合わせフォーム：</strong>https://chord.codes/contact</li>
</ul>
</div>
</div>
</section>
</div>
</div>
</div>
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

<script id="back-to-top">
document.addEventListener('DOMContentLoaded', function() {
const backToTopBtn = document.getElementById('back-to-top');
const scrollToTopBtns = document.querySelectorAll('.scroll-to-top');

if (backToTopBtn) {
backToTopBtn.addEventListener('click', function() {
window.scrollTo({
top: 0,
behavior: 'smooth'
});
});
}

scrollToTopBtns.forEach(btn => {
btn.addEventListener('click', function() {
window.scrollTo({
top: 0,
behavior: 'smooth'
});
});
});
});
</script>

<script id="active-section">
document.addEventListener('DOMContentLoaded', function() {
const sections = document.querySelectorAll('.section-content');
const tocLinks = document.querySelectorAll('.toc-item');

const observer = new IntersectionObserver((entries) => {
entries.forEach(entry => {
if (entry.isIntersecting) {
const id = entry.target.getAttribute('id');
tocLinks.forEach(link => {
link.classList.remove('bg-secondary', 'text-white');
link.classList.add('text-gray-600');
if (link.getAttribute('href') === `#${id}`) {
link.classList.remove('text-gray-600');
link.classList.add('bg-secondary', 'text-white');
}
});
}
});
}, {
threshold: 0.3,
rootMargin: '-20% 0px -70% 0px'
});

sections.forEach(section => {
observer.observe(section);
});
});
</script>
</body>
</html>