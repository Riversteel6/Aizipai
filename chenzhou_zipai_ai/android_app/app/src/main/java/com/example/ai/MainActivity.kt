package com.example.ai

import android.Manifest
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.example.ai.theme.AITheme

class MainActivity : ComponentActivity() {
    private val permissionEpoch = mutableIntStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AITheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    RuntimeScreen(
                        context = this@MainActivity,
                        permissionEpoch = permissionEpoch.intValue,
                        onPermissionChanged = { permissionEpoch.intValue += 1 },
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        permissionEpoch.intValue += 1
        AiAccessibilityService.refreshRuntimeFeedback()
    }
}

@Composable
private fun RuntimeScreen(
    context: Context,
    permissionEpoch: Int,
    onPermissionChanged: () -> Unit,
) {
    var selected by remember { mutableStateOf(ModeStore.load(context)) }
    var runtime by remember { mutableStateOf(RuntimeStatusCenter.snapshot.value) }
    val serviceConfigured = remember(permissionEpoch) { isServiceEnabled(context) }
    var serviceConnected by remember { mutableStateOf(AiAccessibilityService.isConnected()) }
    val notificationsEnabled = remember(permissionEpoch) {
        RuntimeNotifier.notificationsAllowed(context)
    }
    val backgroundProtected = remember(permissionEpoch) {
        BackgroundRunProtection.isEnabled(context)
    }
    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) {
        onPermissionChanged()
        AiAccessibilityService.refreshRuntimeFeedback()
    }

    DisposableEffect(context) {
        val listener: (RuntimeSnapshot) -> Unit = { next ->
            context.mainExecutor.execute {
                runtime = next
                serviceConnected = AiAccessibilityService.isConnected()
            }
        }
        RuntimeStatusCenter.addListener(listener)
        onDispose { RuntimeStatusCenter.removeListener(listener) }
    }

    LaunchedEffect(permissionEpoch) {
        // OPPO may destroy and rebind the accessibility service while its
        // security guide is closing. Re-read the live service after returning
        // from Settings instead of retaining the pre-guide remember value.
        serviceConnected = AiAccessibilityService.isConnected()
        if (serviceConnected) AiAccessibilityService.refreshRuntimeFeedback()
    }

    LaunchedEffect(serviceConnected, runtime.phase) {
        if (serviceConnected && runtime.phase == RuntimePhase.SERVICE_REQUIRED) {
            AiAccessibilityService.refreshRuntimeFeedback()
        }
    }

    val openNotifications = {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            openNotificationSettings(context)
        }
    }
    val saveMode = {
        ModeStore.save(context, selected)
        RuntimeStatusCenter.update(
            phase = if (runtime.running) runtime.phase else RuntimePhase.READY,
            title = if (runtime.running) runtime.title else "模式已保存",
            detail = if (runtime.running) runtime.detail else "可以启动字牌AI",
            mode = selected,
        )
    }

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        if (maxWidth > maxHeight) {
            Row(
                modifier = Modifier.fillMaxSize().padding(horizontal = 32.dp, vertical = 24.dp),
                horizontalArrangement = Arrangement.spacedBy(40.dp),
            ) {
                StatusPanel(
                    runtime = runtime,
                    serviceConfigured = serviceConfigured,
                    serviceConnected = serviceConnected,
                    notificationsEnabled = notificationsEnabled,
                    backgroundProtected = backgroundProtected,
                    openAccessibility = { context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) },
                    openNotifications = openNotifications,
                    openBackgroundProtection = { BackgroundRunProtection.request(context) },
                    modifier = Modifier.weight(1f),
                )
                ControlPanel(
                    selected = selected,
                    onSelect = { selected = it },
                    serviceConnected = serviceConnected,
                    backgroundProtected = backgroundProtected,
                    runtime = runtime,
                    saveMode = saveMode,
                    modifier = Modifier.weight(1f),
                )
            }
        } else {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                    .padding(horizontal = 24.dp, vertical = 40.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp),
            ) {
                StatusPanel(
                    runtime = runtime,
                    serviceConfigured = serviceConfigured,
                    serviceConnected = serviceConnected,
                    notificationsEnabled = notificationsEnabled,
                    backgroundProtected = backgroundProtected,
                    openAccessibility = { context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) },
                    openNotifications = openNotifications,
                    openBackgroundProtection = { BackgroundRunProtection.request(context) },
                )
                HorizontalDivider()
                ControlPanel(
                    selected = selected,
                    onSelect = { selected = it },
                    serviceConnected = serviceConnected,
                    backgroundProtected = backgroundProtected,
                    runtime = runtime,
                    saveMode = saveMode,
                )
            }
        }
    }
}

@Composable
private fun StatusPanel(
    runtime: RuntimeSnapshot,
    serviceConfigured: Boolean,
    serviceConnected: Boolean,
    notificationsEnabled: Boolean,
    backgroundProtected: Boolean,
    openAccessibility: () -> Unit,
    openNotifications: () -> Unit,
    openBackgroundProtection: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("字牌AI", style = MaterialTheme.typography.headlineMedium)
        Text(stringResource(R.string.app_version_label), style = MaterialTheme.typography.labelLarge)
        Text(runtime.title, style = MaterialTheme.typography.titleMedium)
        Text(runtime.detail, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(4.dp))
        PermissionRow(
            label = stringResource(R.string.accessibility_permission),
            status = when {
                serviceConnected -> stringResource(R.string.permission_enabled)
                serviceConfigured -> stringResource(R.string.permission_reconnect)
                else -> stringResource(R.string.permission_disabled)
            },
            actionVisible = !serviceConnected,
            onOpen = openAccessibility,
        )
        PermissionRow(
            label = stringResource(R.string.notification_permission),
            status = if (notificationsEnabled) stringResource(R.string.permission_enabled)
            else stringResource(R.string.permission_disabled),
            actionVisible = !notificationsEnabled,
            onOpen = openNotifications,
        )
        PermissionRow(
            label = stringResource(R.string.background_permission),
            status = if (backgroundProtected) stringResource(R.string.permission_enabled)
            else stringResource(R.string.permission_disabled),
            actionVisible = !backgroundProtected,
            onOpen = openBackgroundProtection,
        )
    }
}

@Composable
private fun ControlPanel(
    selected: GameMode,
    onSelect: (GameMode) -> Unit,
    serviceConnected: Boolean,
    backgroundProtected: Boolean,
    runtime: RuntimeSnapshot,
    saveMode: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ModeRow(
            label = stringResource(R.string.mode_no_wang),
            selected = selected == GameMode.NO_WANG,
            onSelect = { onSelect(GameMode.NO_WANG) },
        )
        ModeRow(
            label = stringResource(R.string.mode_wang),
            selected = selected == GameMode.WANG,
            onSelect = { onSelect(GameMode.WANG) },
        )
        Button(onClick = saveMode, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text(stringResource(R.string.save))
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Button(
                onClick = {
                    saveMode()
                    AiAccessibilityService.startRuntime()
                },
                enabled = serviceConnected && backgroundProtected && !runtime.running,
                modifier = Modifier.weight(1f).height(52.dp),
            ) { Text(stringResource(R.string.start_runtime)) }
            OutlinedButton(
                onClick = { AiAccessibilityService.stopRuntime() },
                enabled = serviceConnected,
                modifier = Modifier.weight(1f).height(52.dp),
            ) { Text(stringResource(R.string.stop_runtime)) }
        }
        Text(
            stringResource(R.string.volume_hint),
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.align(Alignment.CenterHorizontally),
        )
    }
}

@Composable
private fun PermissionRow(
    label: String,
    status: String,
    actionVisible: Boolean,
    onOpen: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().height(52.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
        Text(
            status,
            style = MaterialTheme.typography.bodyMedium,
        )
        if (actionVisible) {
            OutlinedButton(onClick = onOpen, modifier = Modifier.padding(start = 12.dp)) {
                Text(stringResource(R.string.open_settings))
            }
        }
    }
}

@Composable
private fun ModeRow(label: String, selected: Boolean, onSelect: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().height(52.dp).clickable(onClick = onSelect),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = onSelect)
        Text(text = label, style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(start = 12.dp))
    }
}

internal fun isServiceEnabled(context: Context): Boolean {
    val configured = Settings.Secure.getString(
        context.contentResolver,
        Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
    )
    return containsAccessibilityService(
        configured,
        expectedPackage = context.packageName,
        expectedClass = AiAccessibilityService::class.java.name,
    )
}

private fun openNotificationSettings(context: Context) {
    context.startActivity(
        Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
            .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName),
    )
}

internal fun containsAccessibilityService(
    configured: String?,
    expectedPackage: String,
    expectedClass: String,
): Boolean = configured
    ?.split(':')
    ?.any { entry ->
        val separator = entry.indexOf('/')
        if (separator <= 0 || separator == entry.lastIndex) return@any false
        val packageName = entry.substring(0, separator)
        val rawClass = entry.substring(separator + 1)
        val className = if (rawClass.startsWith('.')) packageName + rawClass else rawClass
        packageName.equals(expectedPackage, ignoreCase = true) &&
            className.equals(expectedClass, ignoreCase = true)
    } == true
