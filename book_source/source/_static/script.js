//  Add autoplay functionality to start video after it is in view
// Source: https://stackoverflow.com/questions/32554260/html5-video-run-when-completely-visible

(function() {
    function isScrolledIntoView(element) {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;

        return elementTop >= 0 && elementBottom <= window.innerHeight;
    }

    function playVideosInView(videos) {
        videos.forEach(function(video) {
            if (isScrolledIntoView(video)) {
                video.muted = true;
                video.play();
            } else {
                video.pause();
            }
        });
    }

    window.onload = function () {
        const videos = document.querySelectorAll("video");

        // Play any videos currently in view
        playVideosInView(videos);

        // Every time we scroll, play/pause videos in/out of view
        window.addEventListener("scroll", function() {
            playVideosInView(videos);
        });
    }
})();
