import java.util.Properties

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
  id("com.chaquo.python")
}

val releaseKeystoreProperties = Properties().apply {
    val source = rootProject.file("keystore.properties")
    if (source.isFile) source.inputStream().use { load(it) }
}
val releaseSigningReady = !releaseKeystoreProperties.isEmpty
val appVersionName = "1.0"

android {
    namespace = "com.example.ai"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.example.ai"
        minSdk = 30
        targetSdk = 36
        versionCode = 73
        versionName = appVersionName
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
            if (providers.gradleProperty("aizipaiEmulatorAbi").orNull == "true") {
                abiFilters += "x86_64"
            }
        }
    }

    if (releaseSigningReady) {
        signingConfigs.create("release") {
            storeFile = rootProject.file(releaseKeystoreProperties.getProperty("storeFile"))
            storePassword = releaseKeystoreProperties.getProperty("storePassword")
            keyAlias = releaseKeystoreProperties.getProperty("keyAlias")
            keyPassword = releaseKeystoreProperties.getProperty("keyPassword")
        }
    }
    buildTypes {
        debug {
            if (releaseSigningReady) signingConfig = signingConfigs.getByName("release")
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            if (releaseSigningReady) signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
      compose = true
      aidl = false
      buildConfig = false
      shaders = false
    }

    packaging {
      resources {
        excludes += "/META-INF/{AL2.0,LGPL2.1}"
      }
    }
}

chaquopy {
    defaultConfig {
        version = "3.10"
        buildPython(System.getenv("AIZIPAI_PYTHON") ?: "python")
        pip {
            install("numpy==1.26.2")
            install("opencv-python==4.5.1.48")
            install("Pillow==11.0.0")
            install("PyYAML==6.0.3")
        }
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)
  androidTestImplementation(composeBom)

  // Core Android dependencies
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")

  // Arch Components
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  // Compose
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  // Tooling
  debugImplementation(libs.androidx.compose.ui.tooling)
  // Instrumented tests
  androidTestImplementation(libs.androidx.compose.ui.test.junit4)
  debugImplementation(libs.androidx.compose.ui.test.manifest)

  // Local tests: jUnit, coroutines, Android runner
  testImplementation(libs.junit)
  testImplementation(libs.kotlinx.coroutines.test)
  testImplementation("org.json:json:20250517")

  // Instrumented tests: jUnit rules and runners
  androidTestImplementation(libs.androidx.test.core)
  androidTestImplementation(libs.androidx.test.ext.junit)
  androidTestImplementation(libs.androidx.test.runner)
  androidTestImplementation(libs.androidx.test.espresso.core)

  // Navigation
  implementation(libs.androidx.navigation3.ui)
  implementation(libs.androidx.navigation3.runtime)
  implementation(libs.androidx.lifecycle.viewmodel.navigation3)
}

val prepareMobilePython by tasks.registering(Exec::class) {
    commandLine(
        "C:/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe",
        rootProject.file("prepare_mobile_runtime.py").absolutePath,
    )
}

tasks.matching { it.name.endsWith("PythonSources") }.configureEach {
    dependsOn(prepareMobilePython)
}
