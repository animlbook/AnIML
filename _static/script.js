//  Add autoplay functionality to start video after it is in view
// Source: https://stackoverflow.com/questions/32554260/html5-video-run-when-completely-visible

(function() {
    function stripHtml(html) {
       let tmp = document.createElement("DIV");
       tmp.innerHTML = html;
       return tmp.textContent || tmp.innerText || "";
    }

    function isScrolledIntoView(element) {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;

        return elementTop >= 0 && elementBottom <= window.innerHeight;
    }

    function playVideosInView(videos) {
        videos.forEach(function(video) {
            // Only autoplay for videos that haven't been autoplayed originally
            if (!video.hasAttribute("hasPlayed")) {
                if (isScrolledIntoView(video)) {
                    video.muted = true;
                    video.play();
                    video.setAttribute('hasPlayed', true);
                } else {
                    video.pause();
                }
            }
        });
    }

    window.onload = function () {
        // Remove HTML from page title D:
        document.title = stripHtml(document.title);


        // Control video autoplay
        const videos = document.querySelectorAll("video");

        // Play any videos currently in view
        playVideosInView(videos);

        // Every time we scroll, play/pause videos in/out of view
        window.addEventListener("scroll", function() {
            playVideosInView(videos);
        });

        // Remove controls from videos unless hoverd
        $("video").hover(function(event) {
            if(event.type === "mouseenter") {
                $(this).attr("controls", "");
            } else if(event.type === "mouseleave") {
                $(this).removeAttr("controls");
            }
        });
    }
})();
